"""Structured slot diffusion: corruption, mean-field reverse updates, reference sampler.

AGENTS.md 3.2-3.3.

Slot codes: ``YES = 0``, ``NO = 1``, ``MASK = 2``. A soft state ``mu`` has
shape ``[K, 3]`` with columns (Yes, No, Mask) and rows summing to one.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

YES, NO, MASK = 0, 1, 2


def sample_t(gen: torch.Generator, full_mask_prob: float = 0.5, t_min: float = 1e-3) -> float:
    """Half of corruption draws use t = 1; else t ~ U[t_min, 1)."""
    if float(torch.rand((), generator=gen)) < full_mask_prob:
        return 1.0
    return float(t_min + (1.0 - t_min) * torch.rand((), generator=gen))


def sample_clean_action(target: torch.Tensor, gen: torch.Generator) -> int:
    """Sample one legal clean action from the (hard or soft) target over [ref, 1..K].

    Returns 0 for the reference, i for candidate i.
    """
    p = target.detach().double().cpu()
    p = p / p.sum()
    return int(torch.multinomial(p, 1, generator=gen))


def clean_labels(action: int, K: int, device=None) -> torch.Tensor:
    lab = torch.full((K,), NO, dtype=torch.long, device=device)
    if action > 0:
        lab[action - 1] = YES
    return lab


def corrupt(labels: torch.Tensor, t: float, noise: torch.Tensor) -> torch.Tensor:
    """Replace each slot label by MASK independently with probability t.

    ``noise`` is an explicit uniform ``[K]`` tensor so that the corruption can be
    coupled across candidate permutations (permute ``noise`` together with the
    candidates). Query and option text are never corrupted.
    """
    out = labels.clone()
    out[noise < t] = MASK
    return out


def hard_to_soft(state: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    return torch.nn.functional.one_hot(state, 3).to(dtype)


def initial_soft_state(K: int, dtype=torch.float32, device=None) -> torch.Tensor:
    mu = torch.zeros(K, 3, dtype=dtype, device=device)
    mu[:, MASK] = 1.0
    return mu


def mean_field_update(mu: torch.Tensor, p: torch.Tensor, t: float, s: float) -> torch.Tensor:
    """Deterministic transition t -> s using current-pass probabilities ``p [K]``.

    beta = 1 - s/t;  Yes += beta*Mask*P;  No += beta*Mask*(1-P);  Mask *= s/t.
    All updates use the same old state. No hard compatibility constraints.
    """
    if not (0.0 < s < t <= 1.0):
        raise ValueError(f"need 0 < s < t <= 1, got t={t}, s={s}")
    beta = 1.0 - s / t
    m = mu[:, MASK]
    p = p.to(mu.dtype)
    new = torch.empty_like(mu)
    new[:, YES] = mu[:, YES] + beta * m * p
    new[:, NO] = mu[:, NO] + beta * m * (1.0 - p)
    new[:, MASK] = m * (s / t)
    return new


def slot_inputs(
    h_base: torch.Tensor,
    mu: torch.Tensor,
    e_yes: torch.Tensor,
    e_no: torch.Tensor,
    e_mask: torch.Tensor,
) -> torch.Tensor:
    """Boundary feedback, reset from ``h_base`` before every evaluation:

    h_i = h_i_base + mu_Y E(Yes) + mu_N E(No) + mu_M E(Mask) - E(Mask)
    """
    E = torch.stack([e_yes, e_no, e_mask], dim=0).to(h_base.dtype)  # [3, d]
    return h_base + mu.to(h_base.dtype) @ E - e_mask.to(h_base.dtype)


def stochastic_reference_step(
    state: torch.Tensor,
    log_p_compat: torch.Tensor,
    beta: float,
    gen: torch.Generator,
) -> Tuple[torch.Tensor, int]:
    """Reference stochastic sampler step (kept separate from default evaluation).

    Samples ONE compatible legal action from the constrained distribution
    ``log_p_compat [K + 1]``, then reveals each still-masked coordinate
    independently with probability ``beta`` according to that action. Never
    samples competing Yes actions independently.
    """
    p = log_p_compat.detach().double().exp().cpu()
    p = p / p.sum()
    a = int(torch.multinomial(p, 1, generator=gen))
    K = state.shape[0]
    target = clean_labels(a, K, device=state.device)
    reveal = (torch.rand(K, generator=gen) < beta).to(state.device) & (state == MASK)
    new = state.clone()
    new[reveal] = target[reveal]
    return new, a


def is_legal_hard_state(state: torch.Tensor) -> bool:
    return int((state == YES).sum()) <= 1


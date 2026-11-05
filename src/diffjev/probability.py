"""Structured slot partition normalisation (AGENTS.md 3.1) and task readouts (4).

For K explicit slots the legal clean outcomes are a virtual reference (all
No; fixed energy 0) and candidates 1..K (exactly one Yes). With
``z_i = logit_i(Yes) - logit_i(No)``:

    P_ref = 1 / (1 + sum_j exp(z_j)),   P_i = exp(z_i) * P_ref

computed with ``logsumexp`` over energies ``[0, z_1, ..., z_K]``. For a hard
corrupted state, outcomes incompatible with revealed labels are excluded
before normalisation. The Bernoulli marginal of slot i is ``r_i = P_i``.

Outcome index convention: column 0 = reference, column i = candidate i.
"""

from __future__ import annotations

from typing import Optional

import torch

from .diffusion import MASK, NO, YES

NEG = -1.0e9  # finite "minus infinity" keeps gradients finite at boundaries


def compatibility_mask(hard_state: torch.Tensor) -> torch.Tensor:
    """Compatible clean outcomes ``[K + 1]`` (bool) for a hard state ``[K]``.

    Codes: YES, NO, MASK. Raises if the state is illegal (two revealed Yes).
    """
    K = hard_state.shape[-1]
    yes = hard_state == YES
    n_yes = int(yes.sum())
    out = torch.ones(K + 1, dtype=torch.bool, device=hard_state.device)
    if n_yes > 1:
        raise ValueError("illegal hard state: more than one revealed Yes")
    if n_yes == 1:
        out[:] = False
        out[1 + int(torch.nonzero(yes)[0])] = True
        return out
    out[1:] = hard_state != NO
    return out


def energies(
    z: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    compat: Optional[torch.Tensor] = None,
    include_reference: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Energies ``[..., K + 1]``: ``[0, z_1/T, ..., z_K/T]`` with exclusions.

    ``valid``: ``[..., K]`` bool for padded candidate sets.
    ``compat``: ``[..., K + 1]`` bool compatibility with a hard state.
    ``include_reference=False`` gives the forced-choice adapter (reference
    excluded from normalisation).
    """
    z = z / temperature
    e = torch.cat([torch.zeros_like(z[..., :1]), z], dim=-1)
    keep = torch.ones_like(e, dtype=torch.bool)
    if valid is not None:
        keep[..., 1:] &= valid.bool()
    if compat is not None:
        keep &= compat.bool()
    if not include_reference:
        keep[..., 0] = False
    return torch.where(keep, e, torch.full_like(e, NEG))


def structured_log_probs(
    z: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    compat: Optional[torch.Tensor] = None,
    include_reference: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log-probabilities over ``[ref, 1..K]`` (float32 or wider)."""
    work = z.float() if z.dtype in (torch.float16, torch.bfloat16) else z
    e = energies(work, valid, compat, include_reference, temperature)
    return e - torch.logsumexp(e, dim=-1, keepdim=True)


def log_one_minus_marginals(log_p: torch.Tensor) -> torch.Tensor:
    """Exact ``log(1 - P_i)`` for candidates (shape ``[..., K]``).

    ``log(1 - P_i) = logsumexp_{a != i} log P_a`` computed without
    cancellation. O(K^2) memory; used in training (K <= 32 by default).
    """
    K1 = log_p.shape[-1]
    K = K1 - 1
    expanded = log_p.unsqueeze(-2).expand(*log_p.shape[:-1], K, K1).clone()
    idx = torch.arange(K, device=log_p.device)
    expanded[..., idx, idx + 1] = NEG
    return torch.logsumexp(expanded, dim=-1)


# ----------------------------------------------------------------------------
# Task readouts
# ----------------------------------------------------------------------------


def forced_choice_log_probs(z: torch.Tensor, valid: Optional[torch.Tensor] = None, temperature: float = 1.0):
    """Benchmarks requiring a forced choice: softmax over candidates only.

    Returns ``[..., K]`` (no reference column)."""
    lp = structured_log_probs(z, valid, None, include_reference=False, temperature=temperature)
    return lp[..., 1:]


def noul_probability(z1: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """``P(Yes) = sigmoid(z_1)`` for the single positive proposition."""
    return torch.sigmoid(z1 / temperature)


def score_readout(log_p: torch.Tensor, bin_values: torch.Tensor):
    """Score mode: bins sorted ascending; column 0 = lowest bin (reference).

    Returns (probs ``[M]``, expected score, CDF ``[M]``). The CDF is obtained by
    cumulative sums of the interval-bin distribution.
    """
    p = log_p.exp()
    if bin_values.shape[-1] != p.shape[-1]:
        raise ValueError("bin_values must have M = K + 1 entries")
    if not bool((bin_values[1:] > bin_values[:-1]).all()):
        raise ValueError("score bin values must be strictly increasing")
    expected = (p * bin_values.to(p.dtype)).sum(-1)
    cdf = torch.cumsum(p, dim=-1)
    return p, expected, cdf


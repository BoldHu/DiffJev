"""Training objectives (AGENTS.md 3.2).

Default objective: ``L_mask + mean_passes(L_task_CE) + 0.1 * L_Brier``.
Targets are distributions over ``[ref, 1..K]`` (hard = one-hot, soft kept).
"""

from __future__ import annotations

import torch

from .diffusion import MASK, YES
from .probability import log_one_minus_marginals


def mask_loss(log_p_compat: torch.Tensor, labels: torch.Tensor, state: torch.Tensor, t: float) -> torch.Tensor:
    """``sum_i 1[x_t_i == MASK] * BCE(y_i, P_i) / (K * t)``.

    ``log_p_compat``: structured log-probs under the hard-state compatibility
    constraint, ``[K + 1]``. ``labels``: clean slot labels ``[K]``. The
    denominator is K (not the number of masked slots); with no masked slot
    the loss is exactly zero. BCE is evaluated in log space (stable at P in
    {0, 1}).
    """
    K = labels.shape[0]
    log_r = log_p_compat[1:]
    log_1mr = log_one_minus_marginals(log_p_compat)
    y = (labels == YES)
    nll = -torch.where(y, log_r, log_1mr)
    masked = (state == MASK).to(nll.dtype)
    return (masked * nll).sum() / (K * t)


def task_ce(log_p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structured cross-entropy ``-sum_a target_a log P_a`` (hard or soft)."""
    target = target.to(log_p.dtype)
    return -(target * log_p).sum(-1)


def brier(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Categorical Brier ``sum_a (P_a - target_a)^2``."""
    return ((p - target.to(p.dtype)) ** 2).sum(-1)


def cumulative_brier(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cumulative Brier for ordered score bins (optional replacement)."""
    return ((torch.cumsum(p, -1) - torch.cumsum(target.to(p.dtype), -1)) ** 2).sum(-1)

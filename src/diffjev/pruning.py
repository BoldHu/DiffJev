"""Progressive candidate pruning (AGENTS.md 3.4).

- Disabled for K <= 32, ``noul``, ``score`` and full-support runs.
- Survivors are selected by current candidate energies; all exact ties at the
  cutoff are retained (the nominal cap may be exceeded).
- Required single-choice tie breaks use immutable semantic identity, never
  array position or display aliases.
- Gold candidates are never rescued; gold survival is only recorded.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def pruning_active(K: int, mode: str, enabled: bool, min_k: int, full_support: bool) -> bool:
    return bool(enabled and not full_support and mode == "choice" and K > min_k)


def select_survivors(z: torch.Tensor, cap: int) -> torch.Tensor:
    """Indices (into ``z``) of survivors, ascending order; ties at cutoff kept."""
    s = z.shape[0]
    if cap >= s:
        return torch.arange(s, device=z.device)
    cutoff = torch.topk(z, cap).values.min()
    keep = torch.nonzero(z >= cutoff, as_tuple=False).flatten()
    return keep


def top1_by_identity(scores: torch.Tensor, candidate_ids: Sequence[str]) -> str:
    """Single-choice argmax; exact ties broken by the smallest semantic id."""
    best = scores.max()
    tied = [candidate_ids[i] for i in torch.nonzero(scores == best, as_tuple=False).flatten().tolist()]
    return min(tied)


def gold_survived(active_ids: Sequence[str], gold_id: Optional[str]) -> Optional[bool]:
    if gold_id is None:
        return None
    return gold_id in set(active_ids)


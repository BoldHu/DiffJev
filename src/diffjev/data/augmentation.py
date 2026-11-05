"""Candidate-count augmentation and null augmentation (AGENTS.md 5.2).

- Eligible hard single-positive choice training: propose
  K in {2, 4, 8, 16, 32} with probs {0.50, 0.25, 0.15, 0.07, 0.03}.
  Original menus are preserved: renormalise over target sizes >= native K.
- Negatives come from a versioned training-partition bank with relevant
  schema/task meaning (the caller supplies the eligible bank).
- Null augmentation (15% of eligible examples) removes the positive only when
  the caller certifies that no remaining candidate is valid.
- Never applied to soft-label, noul or score tasks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_K_PROPOSAL: Dict[int, float] = {2: 0.50, 4: 0.25, 8: 0.15, 16: 0.07, 32: 0.03}


def sample_target_k(native_k: int, rng: random.Random, proposal: Optional[Dict[int, float]] = None) -> int:
    proposal = proposal or DEFAULT_K_PROPOSAL
    sizes = [k for k in sorted(proposal) if k >= native_k]
    if not sizes:
        return native_k  # native menu already exceeds the largest proposal; keep as is
    w = [proposal[k] for k in sizes]
    return rng.choices(sizes, weights=w, k=1)[0]


@dataclass
class AugmentedMenu:
    candidate_ids: List[str]
    target_id: Optional[str]  # None = reference ("none of these")
    native_k: int
    realized_k: int
    proposed_k: int
    null_augmented: bool


def augment_menu(
    menu_ids: Sequence[str],
    positive_id: str,
    negative_bank: Sequence[str],
    rng: random.Random,
    proposal: Optional[Dict[int, float]] = None,
    null_rate: float = 0.15,
    null_certified: bool = False,
    max_k: int = 32,
) -> AugmentedMenu:
    """Extend a preserved menu with unique bank negatives; optionally null-augment."""
    if positive_id not in menu_ids:
        raise ValueError("positive must be in the native menu")
    if len(set(menu_ids)) != len(menu_ids):
        raise ValueError("native menu contains duplicate candidate IDs")
    native = len(menu_ids)
    if native > max_k:
        raise ValueError("examples with native K > max_k are filtered, not cropped")
    k = sample_target_k(native, rng, proposal)
    existing = set(menu_ids)
    pool = list(dict.fromkeys(c for c in negative_bank if c not in existing and c != positive_id))
    rng.shuffle(pool)
    extra = pool[: max(0, k - native)]
    ids = list(menu_ids) + extra
    target: Optional[str] = positive_id
    null_aug = False
    if null_certified and rng.random() < null_rate and len(ids) > 1:
        ids = [c for c in ids if c != positive_id]
        target = None
        null_aug = True
    rng.shuffle(ids)
    return AugmentedMenu(ids, target, native, len(ids), k, null_aug)


"""Grouped, deterministic splitting and deduplication (AGENTS.md 5.2).

Group duplicate source problems/templates across sources (reordered twins and
near-duplicate queries share a group), then assign groups to
train/development/calibration with 90/5/5 by a stable hash. Records whose
group overlaps a frozen evaluation family are quarantined, never trained on.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SPLITS = ("train", "dev", "calib")


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def assign_split(group_key: str, ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05), salt: str = "diffjev-v1") -> str:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    u = (stable_hash(f"{salt}:{group_key}") % 10**8) / 10**8
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "dev"
    return "calib"


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_records(records: Sequence[Dict], keys_fn: Callable[[Dict], Iterable[str]]) -> Dict[str, str]:
    """Union records sharing any key (stable id, normalized query text, template id).

    Each record must have ``"rid"``. Returns rid -> group id.
    """
    uf = UnionFind()
    for r in records:
        rid = f"rid:{r['rid']}"
        uf.find(rid)
        for k in keys_fn(r):
            uf.union(rid, f"key:{k}")
    return {r["rid"]: uf.find(f"rid:{r['rid']}") for r in records}


def split_records(
    records: Sequence[Dict],
    keys_fn: Callable[[Dict], Iterable[str]],
    eval_keys: Optional[Set[str]] = None,
    ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05),
) -> Dict[str, List[str]]:
    """Return {"train": [...], "dev": [...], "calib": [...], "quarantine": [...]}"""
    eval_keys = eval_keys or set()
    groups = group_records(records, keys_fn)
    tainted: Set[str] = set()
    for r in records:
        if any(k in eval_keys for k in keys_fn(r)):
            tainted.add(groups[r["rid"]])
    out: Dict[str, List[str]] = {s: [] for s in SPLITS}
    out["quarantine"] = []
    for r in records:
        g = groups[r["rid"]]
        if g in tainted:
            out["quarantine"].append(r["rid"])
        else:
            out[assign_split(g, ratios)].append(r["rid"])
    return out


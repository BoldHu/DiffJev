"""Task adapters (AGENTS.md section 4).

| Mode   | Explicit slots            | Reference          | Returned value                         |
|--------|---------------------------|--------------------|----------------------------------------|
| choice | K semantic alternatives   | None of these      | candidate probs + reference prob       |
| noul   | one positive proposition  | No                 | P(Yes) = sigmoid(z_1) and complement   |
| score  | M - 1 ordered interval bins | lowest score bin | all M bin probs + expected score       |

Targets are distributions over ``[ref, cand_1, ..., cand_K]``; soft targets
are preserved as given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .packing import Candidate

REFERENCE_ID = "__reference__"
MODES = ("choice", "noul", "score")


@dataclass
class DecisionExample:
    mode: str
    query_ids: List[int]
    candidates: List[Candidate]
    target: Optional[torch.Tensor] = None  # [K + 1] over [ref, cands]
    bin_values: Optional[List[float]] = None  # score only: M values ascending, [ref, cands]
    forced_choice: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        ids = [c.cid for c in self.candidates]
        if len(ids) != len(set(ids)) or REFERENCE_ID in ids:
            raise ValueError("candidate IDs must be unique and distinct from the reference")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.target is not None:
            if self.target.shape[-1] != len(self.candidates) + 1:
                raise ValueError("target must cover [ref] + candidates")
            s = float(self.target.sum())
            if abs(s - 1.0) > 1e-5 or bool((self.target < 0).any()):
                raise ValueError("target must be a probability distribution")
        if self.mode == "noul" and len(self.candidates) != 1:
            raise ValueError("noul uses exactly one explicit slot")
        if self.mode == "score":
            if self.bin_values is None or len(self.bin_values) != len(self.candidates) + 1:
                raise ValueError("score needs M = K + 1 bin values")
            if not all(a < b for a, b in zip(self.bin_values[:-1], self.bin_values[1:])):
                raise ValueError("score bins must be sorted strictly ascending (lowest = reference)")

    @property
    def gold_id(self) -> Optional[str]:
        """Candidate id with all target mass (hard single-positive), else None."""
        if self.target is None:
            return None
        idx = int(torch.argmax(self.target))
        if float(self.target[idx]) < 1.0 - 1e-9:
            return None
        return REFERENCE_ID if idx == 0 else self.candidates[idx - 1].cid

    def permuted(self, perm: Sequence[int]) -> "DecisionExample":
        """Reorder candidates by ``perm`` (semantics, ids and target follow)."""
        perm = list(perm)
        cands = [self.candidates[i] for i in perm]
        target = None
        if self.target is not None:
            target = torch.cat([self.target[:1], self.target[1:][torch.tensor(perm)]])
        bins = None
        if self.bin_values is not None:
            raise ValueError("score bins are ordered; permutation tests apply to choice")
        return DecisionExample(self.mode, list(self.query_ids), cands, target, bins, self.forced_choice, dict(self.meta))


def _enc(tokenizer, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def build_choice(
    tokenizer,
    query_text: str,
    options: Sequence[Tuple[str, str]],
    target: Optional[Dict[str, float]] = None,
    forced_choice: bool = False,
    aliases: Optional[Sequence[str]] = None,
) -> DecisionExample:
    """``options``: (stable id, semantic text). ``target``: id -> prob; the key
    ``REFERENCE_ID`` carries "none of these" mass. Aliases are metadata only."""
    cands = [
        Candidate(cid, _enc(tokenizer, text), aliases[i] if aliases else None, text)
        for i, (cid, text) in enumerate(options)
    ]
    t = None
    if target is not None:
        t = torch.zeros(len(cands) + 1, dtype=torch.float64)
        index = {c.cid: i + 1 for i, c in enumerate(cands)}
        for k, v in target.items():
            t[0 if k == REFERENCE_ID else index[k]] += v
    return DecisionExample("choice", _enc(tokenizer, query_text), cands, t, None, forced_choice,
                           {"query_text": query_text})


def build_noul(tokenizer, query_text: str, proposition_text: str, p_yes: Optional[float] = None) -> DecisionExample:
    """One positive proposition; reference = No (label polarity preserved)."""
    cands = [Candidate("yes", _enc(tokenizer, proposition_text), text=proposition_text)]
    t = None if p_yes is None else torch.tensor([1.0 - p_yes, p_yes], dtype=torch.float64)
    return DecisionExample("noul", _enc(tokenizer, query_text), cands, t,
                           meta={"query_text": query_text, "proposition": proposition_text})


def build_score(
    tokenizer,
    rubric_text: str,
    query_text: str,
    bins: Sequence[Tuple[float, str]],
    target: Optional[Dict[float, float]] = None,
) -> DecisionExample:
    """``bins``: (numeric value, bin description) in any order. Sorted by value;
    the lowest bin is the reference (not abstention). The full rubric,
    including the reference-bin definition, goes into the shared query."""
    bins = sorted(bins, key=lambda b: float(b[0]))
    values = [float(b[0]) for b in bins]
    ref_desc = bins[0][1]
    definitions = "\n".join(f"{v:g}: {d}" for v, d in bins)
    query = f"{rubric_text}\nScore bins (lowest is reference):\n{definitions}\n{query_text}".strip()
    cands = [Candidate(f"bin:{v:g}", _enc(tokenizer, d), text=d) for v, d in bins[1:]]
    t = None
    if target is not None:
        t = torch.zeros(len(bins), dtype=torch.float64)
        for v, p in target.items():
            t[values.index(float(v))] += p
    return DecisionExample("score", _enc(tokenizer, query), cands, t, values,
                           meta={"query_text": query, "bins": bins, "rubric": rubric_text})


def merge_null_equivalent(example: DecisionExample, null_ids: Sequence[str]) -> DecisionExample:
    """Merge explicitly flagged null-equivalent options into the reference.

    Requires an explicit, audited flag per option; never inferred from text
    containing "No"."""
    null = set(null_ids)
    keep = [i for i, c in enumerate(example.candidates) if c.cid not in null]
    if len(keep) == len(example.candidates):
        return example
    if not keep:
        raise ValueError("cannot remove every explicit slot")
    t = None
    if example.target is not None:
        drop = [i for i, c in enumerate(example.candidates) if c.cid in null]
        ref = example.target[0] + sum(example.target[1 + i] for i in drop)
        t = torch.cat([ref.reshape(1), example.target[1:][torch.tensor(keep)]])
    cands = [example.candidates[i] for i in keep]
    return DecisionExample(example.mode, list(example.query_ids), cands, t, example.bin_values, example.forced_choice, dict(example.meta))


@dataclass
class DecisionOutput:
    mode: str
    candidate_ids: List[str]  # active (retained) candidates, packed order
    candidate_probs: torch.Tensor  # [s]
    reference_prob: float
    probability_scope: str  # "full_support" | "retained_support"
    z_final: torch.Tensor
    survivor_ids_per_pass: List[List[str]]
    gold_survival: List[Optional[bool]]
    top1: str
    forced_choice_probs: Optional[torch.Tensor] = None
    score_probs: Optional[torch.Tensor] = None
    expected_score: Optional[float] = None
    score_cdf: Optional[torch.Tensor] = None
    p_yes: Optional[float] = None

    def prob_by_id(self) -> Dict[str, float]:
        if self.forced_choice_probs is not None:
            return {cid: float(p) for cid, p in zip(self.candidate_ids, self.forced_choice_probs)}
        d = {cid: float(p) for cid, p in zip(self.candidate_ids, self.candidate_probs)}
        d[REFERENCE_ID] = float(self.reference_prob)
        return d

    def to_record(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "probability_scope": self.probability_scope,
            "normalization": "candidate_only" if self.forced_choice_probs is not None else "reference_and_candidates",
            "active_candidate_ids": list(self.candidate_ids),
            "probs": self.prob_by_id(),
            "top1": self.top1,
            "survivor_ids_per_pass": self.survivor_ids_per_pass,
            "gold_survival": self.gold_survival,
            "expected_score": self.expected_score,
            "p_yes": self.p_yes,
        }

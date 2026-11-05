"""Exp 2 — Option order and external-label robustness (AGENTS.md section 6, Exp 2).

Protocol:
- Evaluates up to 1,000 fixed canonical choice families.
- Enumerates all ``K!`` permutations for ``K <= 4``; otherwise evaluates the
  original order plus at most 10 seeded random permutations.
- Preserves candidate semantics and remaps predictions/probabilities back to
  immutable candidate identities (inverse permutation).
- Evaluates arbitrary display alias renaming (e.g. ``A..D`` -> ``W..Z``,
  ``[1]..[K]``) as a **separate** condition from option reordering, never
  altering meaningful class/tool descriptions.
- Reports semantic Decision Flip Rate (DFR), probability drift after inverse
  permutation, Jensen-Shannon divergence, and accuracy change (mean permuted -
  original, worst-case permuted - original) with canonical-family bootstrap CIs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..baselines.base import BaselineAdapter
from ..model import DiffJevModel
from ..packing import Candidate
from ..tasks import DecisionExample
from .metrics import (
    decision_flip_rate,
    family_bootstrap_ci,
    js_divergence,
    max_probability_drift,
    permutations_for,
)

ALIAS_SCHEMES: Dict[str, List[str]] = {
    "alpha_upper": [chr(ord("A") + i) for i in range(26)],
    "alpha_reverse": [chr(ord("Z") - i) for i in range(26)],
    "numeric_bracket": [f"[{i + 1}]" for i in range(64)],
    "roman_tag": ["(I)", "(II)", "(III)", "(IV)", "(V)", "(VI)", "(VII)", "(VIII)"],
}


def apply_alias_scheme(example: DecisionExample, scheme_name: str) -> DecisionExample:
    """Return a copy of ``example`` with only external display aliases renamed.

    Candidate IDs and semantic token_ids are untouched.
    """
    labels = ALIAS_SCHEMES.get(scheme_name, ALIAS_SCHEMES["numeric_bracket"])
    cands = [
        Candidate(
            cid=c.cid,
            token_ids=list(c.token_ids),
            alias=(labels[i] if i < len(labels) else f"{scheme_name}_{i + 1}"),
            text=c.text,
        )
        for i, c in enumerate(example.candidates)
    ]
    return DecisionExample(
        mode=example.mode,
        query_ids=list(example.query_ids),
        candidates=cands,
        target=example.target.clone() if example.target is not None else None,
        bin_values=list(example.bin_values) if example.bin_values is not None else None,
        forced_choice=example.forced_choice,
        meta=copy.deepcopy(example.meta),
    )


def _predict_canonical(
    runner: Union[DiffJevModel, BaselineAdapter], ex: DecisionExample
) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    if isinstance(runner, DiffJevModel):
        try:
            out = runner.predict(ex, full_support=True)
        except (RuntimeError,ValueError) as error:
            if "context_limit" in str(error) or "out of memory" in str(error).lower():
                return None,None
            raise
        probs = (
            {cid: float(p) for cid, p in zip(out.candidate_ids, out.forced_choice_probs)}
            if (ex.forced_choice and out.forced_choice_probs is not None)
            else out.prob_by_id()
        )
        return out.top1, probs
    res = runner.predict(ex)
    return res.top1, res.prob_by_id


@dataclass
class Exp2RobustnessReport:
    model_name: str
    num_families: int
    order_permutation: Dict[str, Any]
    alias_renaming: Dict[str, Any]
    per_family_records: List[Dict[str, Any]] = field(default_factory=list)


def run_exp2_robustness(
    model_name: str,
    runner: Union[DiffJevModel, BaselineAdapter],
    canonical_families: Sequence[Tuple[str, DecisionExample]],
    max_families: int = 1000,
    n_random_perms: int = 10,
    perm_seed: int = 0,
    alias_schemes: Sequence[str] = ("alpha_upper", "alpha_reverse", "numeric_bracket"),
    n_boot: int = 200,
) -> Exp2RobustnessReport:
    """Run Exp 2 order-permutation and separate alias-renaming robustness tests."""
    selected = [
        (fid, ex) for fid, ex in canonical_families if ex.mode == "choice"
    ][:max_families]

    order_decisions: List[List[str]] = []
    fam_order_flip: Dict[str, float] = {}
    fam_order_drift: Dict[str, float] = {}
    fam_order_js: Dict[str, float] = {}
    fam_orig_acc: Dict[str, float] = {}
    fam_perm_mean_acc: Dict[str, float] = {}
    fam_perm_worst_acc: Dict[str, float] = {}

    alias_decisions: List[List[str]] = []
    fam_alias_flip: Dict[str, float] = {}
    fam_alias_drift: Dict[str, float] = {}

    per_family_records: List[Dict[str, Any]] = []

    for idx, (fid, ex) in enumerate(selected):
        K = len(ex.candidates)
        perms = permutations_for(K, n_random=n_random_perms, seed=perm_seed + idx)
        gold_id = ex.gold_id

        # 1) Pure option-order permutation test (external aliases untouched)
        decs: List[str] = []
        pmaps: List[Dict[str, float]] = []
        for p in perms:
            ex_p = ex.permuted(p)
            top1, probs = _predict_canonical(runner, ex_p)
            decs.append(top1)
            if probs is not None:
                pmaps.append(probs)

        complete = all(d is not None for d in decs)
        if complete:
            order_decisions.append(decs)
        flipped = float(len(set(decs)) > 1) if complete else None
        if complete:
            fam_order_flip[fid] = flipped
        drift = max_probability_drift(pmaps) if len(pmaps) == len(perms) and len(pmaps)>1 else None
        if drift is not None:
            fam_order_drift[fid] = drift
        js_vals = (
            [js_divergence(pmaps[0], pm) for pm in pmaps[1:]] if len(pmaps) > 1 else [0.0]
        )
        if drift is not None:
            fam_order_js[fid] = max(js_vals)

        if gold_id is not None and decs and complete:
            orig_hit = 1.0 if decs[0] == gold_id else 0.0
            perm_hits = [1.0 if d == gold_id else 0.0 for d in decs]
            fam_orig_acc[fid] = orig_hit
            fam_perm_mean_acc[fid] = sum(perm_hits) / len(perm_hits)
            fam_perm_worst_acc[fid] = min(perm_hits)

        # 2) Separate external alias-renaming test (candidate order fixed)
        a_decs: List[str] = []
        a_pmaps: List[Dict[str, float]] = []
        for scheme in alias_schemes:
            ex_a = apply_alias_scheme(ex, scheme)
            top1_a, probs_a = _predict_canonical(runner, ex_a)
            a_decs.append(top1_a)
            if probs_a is not None:
                a_pmaps.append(probs_a)

        alias_complete = all(d is not None for d in a_decs)
        if alias_complete:
            alias_decisions.append(a_decs)
            fam_alias_flip[fid] = float(len(set(a_decs)) > 1)
        if len(a_pmaps) == len(alias_schemes) and len(a_pmaps)>1:
            fam_alias_drift[fid] = max_probability_drift(a_pmaps)

        per_family_records.append(
            {
                "family_id": fid,
                "duplicate_option_groups": ex.meta.get("duplicate_option_groups", []),
                "K": K,
                "num_permutations": len(perms),
                "order_decisions": decs,
                "order_flipped": bool(flipped) if flipped is not None else None,
                "order_complete":complete, "alias_complete":alias_complete,
                "alias_accuracy_change": (sum(d==gold_id for d in a_decs)/len(a_decs)-float(decs[0]==gold_id)) if alias_complete and complete and gold_id is not None else None,
                "order_max_prob_drift": drift,
                "order_max_js": fam_order_js.get(fid),
                "alias_decisions": a_decs,
                "alias_flipped": bool(fam_alias_flip[fid]) if alias_complete else None,
                "alias_max_prob_drift": fam_alias_drift.get(fid),
            }
        )

    orig_acc = sum(fam_orig_acc.values()) / len(fam_orig_acc) if fam_orig_acc else None
    perm_mean_acc = sum(fam_perm_mean_acc.values()) / len(fam_perm_mean_acc) if fam_perm_mean_acc else None
    perm_worst_acc = sum(fam_perm_worst_acc.values()) / len(fam_perm_worst_acc) if fam_perm_worst_acc else None

    order_summary = {
        "decision_flip_rate": decision_flip_rate(order_decisions) if order_decisions else None,
        "complete_family_coverage":len(order_decisions)/max(1,len(selected)),
        "dfr_bootstrap": family_bootstrap_ci(fam_order_flip, n_boot=n_boot),
        "mean_max_prob_drift": sum(fam_order_drift.values()) / len(fam_order_drift) if fam_order_drift else None,
        "max_prob_drift_overall": max(fam_order_drift.values()) if fam_order_drift else None,
        "mean_max_js_divergence": sum(fam_order_js.values()) / len(fam_order_js) if fam_order_js else None,
        "original_accuracy": orig_acc,
        "mean_permuted_accuracy": perm_mean_acc,
        "worst_permuted_accuracy": perm_worst_acc,
        "mean_accuracy_delta": perm_mean_acc - orig_acc if orig_acc is not None else None,
        "worst_accuracy_delta": perm_worst_acc - orig_acc if orig_acc is not None else None,
    }
    alias_summary = {
        "schemes_tested": list(alias_schemes),
        "decision_flip_rate": decision_flip_rate(alias_decisions) if alias_decisions else None,
        "complete_family_coverage":len(alias_decisions)/max(1,len(selected)),
        "dfr_bootstrap": family_bootstrap_ci(fam_alias_flip, n_boot=n_boot),
        "mean_max_prob_drift": sum(fam_alias_drift.values()) / len(fam_alias_drift) if fam_alias_drift else None,
        "max_prob_drift_overall": max(fam_alias_drift.values()) if fam_alias_drift else None,
    }
    alias_deltas = [r["alias_accuracy_change"] for r in per_family_records if r["alias_accuracy_change"] is not None]
    alias_summary["mean_accuracy_delta"] = sum(alias_deltas)/len(alias_deltas) if alias_deltas else None

    return Exp2RobustnessReport(
        model_name=model_name,
        num_families=len(selected),
        order_permutation=order_summary,
        alias_renaming=alias_summary,
        per_family_records=per_family_records,
    )


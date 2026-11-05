"""Exp 1 — Benchmark performance on ``Leanmcp/jevbench`` (AGENTS.md section 6, Exp 1).

Protocol:
- Evaluates the pinned ``Leanmcp/jevbench`` text suite (documented: 12 text
  datasets, 8,016 cases including reordered twins corresponding to 5,516
  canonical families; verifies actual manifest counts rather than assuming
  independence).
- ``SST5`` is evaluated as a ``score`` task.
- Always uses full support (``full_support=True``; no hard pruning) and fixed
  post-hoc temperatures fitted on the internal ``calib`` split (never test-set
  calibration).
- Separates API latency (for ``JevAPIBaseline``) from local GPU latency.
- Reports official aggregate results, per-dataset breakdowns, probability
  metrics (ECE, Brier, NLL) where the model supplies comparable probabilities,
  and canonical-family bootstrap uncertainty.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from ..baselines.base import BaselineAdapter, BaselineResult
from ..model import DiffJevModel
from ..tasks import DecisionExample
from .metrics import (
    brier_score,
    chance_corrected_accuracy,
    expected_calibration_error,
    family_bootstrap_ci,
    hit_at_1,
    negative_log_likelihood,
    percentile,
    score_task_metrics,
)


@dataclass
class JevBenchCase:
    case_id: str
    family_id: str
    dataset: str
    example: DecisionExample
    gold_id: Optional[str] = None
    gold_score: Optional[float] = None


@dataclass
class Exp1ModelReport:
    model_name: str
    num_cases: int
    num_canonical_families: int
    num_datasets: int
    aggregate: Dict[str, Any]
    by_dataset: Dict[str, Dict[str, Any]]
    latency: Dict[str, Optional[float]]
    raw_predictions: List[Dict[str, Any]] = field(default_factory=list)


def _evaluate_single_case(
    runner: Union[DiffJevModel, BaselineAdapter], case: JevBenchCase
) -> Dict[str, Any]:
    ex = case.example
    if isinstance(runner, DiffJevModel):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = runner.predict(ex, full_support=True, gold_id=case.gold_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000.0
        probs = out.prob_by_id()
        return {
            "case_id": case.case_id,
            "family_id": case.family_id,
            "dataset": case.dataset,
            "mode": ex.mode,
            "top1": out.top1,
            "probs": probs,
            "expected_score": out.expected_score,
            "p_yes": out.p_yes,
            "K": len(ex.candidates),
            "probability_scope": out.probability_scope,
            "local_gpu_ms": gpu_ms,
            "api_latency_ms": None,
            "failed": False,
        }
    else:
        res: BaselineResult = runner.predict(ex)
        return {
            "case_id": case.case_id,
            "family_id": case.family_id,
            "dataset": case.dataset,
            "mode": ex.mode,
            "top1": res.top1,
            "probs": res.prob_by_id,
            "expected_score": res.expected_score,
            "p_yes": res.p_yes,
            "K": len(ex.candidates),
            "local_gpu_ms": res.local_gpu_ms,
            "api_latency_ms": res.api_latency_ms,
            "failed": res.failed,
            "failure_reason": res.failure_reason,
        }


def _summarize_group(
    records: Sequence[Dict[str, Any]], cases_by_id: Dict[str, JevBenchCase], n_boot: int = 200
) -> Dict[str, Any]:
    preds: List[Optional[str]] = []
    golds: List[str] = []
    ks: List[int] = []
    prob_maps: List[Dict[str, float]] = []
    prob_golds: List[str] = []
    confidences: List[float] = []
    correct_flags: List[bool] = []
    pred_scores: List[float] = []
    gold_scores: List[float] = []
    fam_hits: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        if r.get("failed"):
            continue
        c = cases_by_id[r["case_id"]]
        if c.example.mode == "score" and c.gold_score is not None and r.get("expected_score") is not None:
            pred_scores.append(float(r["expected_score"]))
            gold_scores.append(float(c.gold_score))
        if c.gold_id is not None:
            preds.append(r["top1"])
            golds.append(c.gold_id)
            ks.append(max(2, len(c.example.candidates)))
            hit = 1.0 if r["top1"] == c.gold_id else 0.0
            fam_hits[c.family_id].append(hit)
            if r.get("probs") is not None:
                pm = r["probs"]
                prob_maps.append(pm)
                prob_golds.append(c.gold_id)
                confidences.append(float(pm.get(r["top1"], 0.0)))
                correct_flags.append(bool(hit == 1.0))

    fam_mean_hit = {f: sum(v) / len(v) for f, v in fam_hits.items()}
    summary: Dict[str, Any] = {
        "num_cases": len(records),
        "num_families": len({r["family_id"] for r in records}),
        "failures": sum(1 for r in records if r.get("failed")),
        "cases_with_duplicate_option_text": sum(bool(cases_by_id[r["case_id"]].example.meta.get("duplicate_option_groups")) for r in records),
        "accuracy": hit_at_1(preds, golds),
        "chance_corrected_accuracy": chance_corrected_accuracy(preds, golds, ks),
        "family_bootstrap_accuracy": family_bootstrap_ci(fam_mean_hit, n_boot=n_boot),
        "accuracy_scope": "successful classification cases; score tasks use their native metrics",
        "coverage": sum(not r.get("failed", False) for r in records)/max(1,len(records)),
    }
    classification = [r for r in records if cases_by_id[r["case_id"]].gold_id is not None]
    summary["accuracy_including_failures"] = (
        sum(not r.get("failed",False) and r.get("top1")==cases_by_id[r["case_id"]].gold_id
            for r in classification)/len(classification) if classification else None
    )
    if prob_maps:
        summary["brier"] = brier_score(prob_maps, prob_golds)
        summary["nll"] = negative_log_likelihood(prob_maps, prob_golds)
        summary["ece"] = expected_calibration_error(confidences, correct_flags)
    else:
        summary["brier"] = None
        summary["nll"] = None
        summary["ece"] = None
    if gold_scores:
        summary["score_metrics"] = score_task_metrics(pred_scores, gold_scores)
    return summary


def run_exp1_benchmark(
    model_name: str,
    runner: Union[DiffJevModel, BaselineAdapter],
    cases: Sequence[JevBenchCase],
    n_boot: int = 200,
    official_output_dir: Optional[str] = None,
) -> Exp1ModelReport:
    """Run Exp 1 JevBench evaluation for a single model or baseline."""
    cases_by_id = {c.case_id: c for c in cases}
    raw: List[Dict[str, Any]] = []
    by_ds_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for c in cases:
        try:
            rec = _evaluate_single_case(runner, c)
        except (RuntimeError, ValueError) as error:
            if "out of memory" not in str(error).lower() and "context_limit" not in str(error):
                raise
            rec = {"case_id":c.case_id,"family_id":c.family_id,"dataset":c.dataset,
                   "mode":c.example.mode,"failed":True,"failure_reason":str(error).split(":")[0],
                   "top1":None,"probs":None,"K":len(c.example.candidates)}
        if c.example.meta.get("duplicate_option_groups"):
            rec["duplicate_option_groups"] = c.example.meta["duplicate_option_groups"]
        raw.append(rec)
        by_ds_records[c.dataset].append(rec)

    by_dataset = {
        ds: _summarize_group(recs, cases_by_id, n_boot=n_boot)
        for ds, recs in sorted(by_ds_records.items())
    }
    aggregate = _summarize_group(raw, cases_by_id, n_boot=n_boot)
    aggregate["coverage"] = sum(not r.get("failed",False) for r in raw)/max(1,len(raw))
    aggregate["summary_scope"] = "project summary; original official task metrics saved separately"
    if official_output_dir is not None:
        from .official import evaluate_official
        aggregate["official"] = evaluate_official(cases, raw, official_output_dir)
    if by_dataset:
        ds_accs = [
            v["accuracy"]
            for v in by_dataset.values()
            if v["accuracy"] is not None and not (isinstance(v["accuracy"], float) and v["accuracy"] != v["accuracy"])
        ]
        aggregate["macro_dataset_accuracy"] = sum(ds_accs) / len(ds_accs) if ds_accs else float("nan")

    gpu_times = [r["local_gpu_ms"] for r in raw if r.get("local_gpu_ms") is not None]
    api_times = [r["api_latency_ms"] for r in raw if r.get("api_latency_ms") is not None]
    latency = {
        "local_gpu_p50_ms": percentile(gpu_times, 50.0) if gpu_times else None,
        "local_gpu_p95_ms": percentile(gpu_times, 95.0) if gpu_times else None,
        "api_p50_ms": percentile(api_times, 50.0) if api_times else None,
        "api_p95_ms": percentile(api_times, 95.0) if api_times else None,
    }

    return Exp1ModelReport(
        model_name=model_name,
        num_cases=len(cases),
        num_canonical_families=len({c.family_id for c in cases}),
        num_datasets=len(by_dataset),
        aggregate=aggregate,
        by_dataset=by_dataset,
        latency=latency,
        raw_predictions=raw,
    )


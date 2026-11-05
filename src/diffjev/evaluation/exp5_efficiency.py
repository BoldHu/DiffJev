"""Exp 5 — Same-size decision latency and computational complexity (AGENTS.md section 6, Exp 5).

Implements both contracts from AGENTS.md lines 208-242:

1. **Timing Contract**:
   - Reuses Exp 3's fixed queries and **BM25-hard negative manifests** across
     ``K in {4, 16, 64, 256, 1024, 2048}`` on the fixed stratified 100-query
     subset.
   - Compares separately adapted **serialized native AR**, **serialized one-pass
     DLM**, and **DiffJev T=3 with pruning**, initialized from the exact same
     checkpoint (plus the LLaDA 8B attribution table with full-context T=1 and
     no-pruning controls).
   - Batch size 1, weights resident, kernels warmed (>= 5 warmups per
     configuration), input-dependent caches cleared before every request,
     within-request cache construction included.
   - 100 queries x 3 repetitions each: aggregates repetitions within a query
     first, then reports query-level ``p50``/``p95`` and query-bootstrap
     uncertainty.
   - Measures complete request wall time from packing/transfer through branch
     encoding, memory prefill, all decision passes, selection, and output
     mapping, synchronizing every participating CUDA device before starting the
     wall timer and after the final GPU-dependent output operation before
     stopping it.
   - Measures GPU kernel time separately via the profiler and records peak
     allocated GPU memory (including weights).
   - Reports unsupported contexts / OOM with explicit ``coverage``; computes
     speedups **only on the intersection of successful queries**, never as
     infinity for a failed baseline.
   - Pairs cost with ``Hit@1`` and stagewise gold survival.

2. **Complexity Contract**:
   - Computes full-request operator-shape GFLOPs (2 FLOPs per multiply-add) and
     theoretical ``Phi(n, a)`` complexity curves using each method's actual
     token counts, padding, and survivor counts ``(K, s, u)``.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..baselines.ar_llm import AutoregressiveJevBaseline
from ..baselines.base import BaselineAdapter
from ..baselines.serialized_dlm import SerializedDLMBaseline
from ..complexity import branch_cost, memory_cost, phi, slot_pass_cost, serialized_dlm_cost, causal_ar_prefill_cost
from ..model import DiffJevModel
from ..tasks import DecisionExample
from ..training import parameter_counts
from .exp3_scaling import DEFAULT_SCALING_KS, ScalingQuerySpec
from .metrics import family_bootstrap_ci, percentile


def _sync_cuda(devices=None) -> None:
    if torch.cuda.is_available():
        for dev_i in (range(torch.cuda.device_count()) if devices is None else devices):
            torch.cuda.synchronize(dev_i)


@torch.inference_mode()
def _measure_single_request(
    runner: Union[DiffJevModel, BaselineAdapter],
    ex: DecisionExample,
    gold_id: str,
    max_option_len: Optional[int],
    max_context_tokens: int,
) -> Dict[str, Any]:
    """Execute one complete fresh request (clearing input-dependent state)."""
    # CPU fixtures can run on CUDA-capable hosts without initializing a GPU.
    # Formal runs synchronize and include weights on every participating device.
    module = runner if hasattr(runner,"parameters") else getattr(runner,"backbone",None)
    devices = sorted({p.device.index for p in module.parameters() if p.is_cuda}) if module is not None else []
    _sync_cuda(devices)
    for index in devices:
        torch.cuda.reset_peak_memory_stats(index)
    if hasattr(runner, "eval"):
        runner.eval()
    if hasattr(runner, "clear_cache"):
        runner.clear_cache()
    _sync_cuda(devices)

    wall_t0 = time.perf_counter()

    try:
        from ..preprocessing import retokenize_example, runner_tokenizer
        if callable(ex):
            ex = ex()
        ex = retokenize_example(ex, runner_tokenizer(runner), max_option_len)
        if isinstance(runner, DiffJevModel):
            packed = runner.pack(ex)
            total_toks = packed.query_len + int(packed.branch_valid.sum().item())
            if int(packed.branch_pos.max()) + 1 > max_context_tokens:
                return {"failed": True, "failure_reason": "context_limit"}
            bd = runner.encode(packed)
            ro = runner.denoiser.rollout(bd, ex.mode, full_support=False)
            out = runner.readout(ex, packed, ro, gold_id=gold_id)
            top1 = out.top1
            surv_counts = [len(s) for s in out.survivor_ids_per_pass]
            final_surv = gold_id in set(out.candidate_ids)
            gold_surv = [bool(s) for s in out.gold_survival]
            pad_toks = int((~packed.branch_valid).sum().item())
        else:
            # For AR in Exp 5 timing, use predict_top1 (1 prefill + cached short decode)
            res = runner.predict_top1(ex, max_context_tokens=max_context_tokens)
            if res.failed:
                return {"failed": True, "failure_reason": res.failure_reason}
            top1 = res.top1
            total_toks = res.total_tokens
            pad_toks = 0
            surv_counts = [len(ex.candidates)]
            gold_surv = [True]
    except (RuntimeError, ValueError) as e:
        if "context_limit" in str(e):
            return {"failed": True, "failure_reason": "context_limit"}
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"failed": True, "failure_reason": "oom"}
        raise

    _sync_cuda(devices)
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    # CUDA events around CPU preprocessing include host idle gaps. Kernel-only
    # time is obtained in a separate profiler run, never substituted with CPU time.
    gpu_ms = float("nan")
    peak_mb = sum(torch.cuda.max_memory_allocated(i) for i in devices) / 2**20

    return {
        "failed": False,
        "wall_ms": wall_ms,
        "gpu_ms": gpu_ms,
        "peak_memory_mb": peak_mb,
        "top1": top1,
        "hit": 1.0 if top1 == gold_id else 0.0,
        "total_tokens": total_toks,
        "padding_tokens": pad_toks,
        "survivor_counts": surv_counts,
        "gold_survival": gold_surv,
        "query_tokens": len(ex.query_ids),
        "option_lengths": [len(c.token_ids) for c in ex.candidates],
        "prompt_tokens": total_toks if isinstance(runner,DiffJevModel) else res.meta.get("prompt_tokens",total_toks),
        "decoded_tokens": 0 if isinstance(runner,DiffJevModel) else res.meta.get("decoded_tokens",0),
    }


def analytic_request_cost(runner, request):
    """Specification Phi units using observed lengths/survivors; not GFLOPs."""
    bb=runner.backbone
    d,L=bb.hidden_size,bb.num_layers
    if isinstance(runner,DiffJevModel):
        q=request["query_tokens"]
        m=max(request["option_lengths"],default=0)
        counts=request["survivor_counts"]
        k=runner.cfg.num_top_layers
        cost=(L-k)*branch_cost(counts[0],q,m,d)
        if len(counts)>1 or runner.cfg.first_pass_detail_memory:
            memory_n=counts[0] if runner.cfg.first_pass_detail_memory else counts[1]
            cost+=k*memory_cost(memory_n,q,m,d)
        for index,n in enumerate(counts):
            detail=int(index>0 or runner.cfg.first_pass_detail_memory)
            pairs=(n*n if runner.cfg.cross_slot_attention else n)+detail*n*(q+m)
            cost+=k*phi(n,pairs,d)
        return cost
    if isinstance(runner,AutoregressiveJevBaseline):
        return causal_ar_prefill_cost(request["prompt_tokens"],d,L,request["decoded_tokens"])
    return serialized_dlm_cost(request["prompt_tokens"],d,L)


def _estimate_method_gflops(
    runner: Union[DiffJevModel, BaselineAdapter],
    ex: DecisionExample,
    survivor_counts: Sequence[int],
) -> Tuple[float, float]:
    """Return ``(operator_shape_gflops, analytic_phi_cost)`` for one request."""
    from .operations import profile_request
    profile = profile_request(lambda: _measure_single_request(runner, ex, ex.gold_id, None,
        getattr(runner, "max_context_tokens", 262144)), cuda=False)
    runner.last_operation_profile = profile
    return profile["gflops"], float("nan")


@dataclass
class Exp5MethodKResult:
    method_name: str
    K: int
    coverage: float
    num_successful_queries: int
    num_total_queries: int
    hit_at_1: float
    wall_p50_ms: float
    wall_p95_ms: float
    wall_p50_bootstrap: Dict[str, float]
    gpu_p50_ms: float
    gpu_p95_ms: float
    peak_memory_mb: float
    mean_gflops: float
    mean_analytic_phi: float
    mean_total_tokens: float
    mean_padding_tokens: float
    stagewise_gold_survival: List[float]
    query_wall_ms: Dict[str, float] = field(default_factory=dict)
    raw_queries: List[Dict[str, Any]] = field(default_factory=list)
    operation_profiles: Dict[str, Any] = field(default_factory=dict)
    wall_p95_bootstrap: Dict[str, float] = field(default_factory=dict)


def time_method_at_k(
    method_name: str,
    runner: Union[DiffJevModel, BaselineAdapter],
    specs: Sequence[ScalingQuerySpec],
    K: int,
    num_queries: int = 100,
    warmups: int = 5,
    repetitions: int = 3,
    max_option_len: Optional[int] = None,
    max_context_tokens: int = 262144,
    n_boot: int = 200,
) -> Exp5MethodKResult:
    """Run the Exp 5 timing contract for one ``(method, K)`` configuration."""
    from .sampling import stratified_subset
    subset = stratified_subset(specs, num_queries)
    examples = [(s.qid, (lambda spec=s: spec.build_example(K, pool="hard")), s.positive.cid) for s in subset]
    if not examples:
        raise ValueError("Timing requires evaluation queries")
    raw_queries, profiles = [], {}

    # Warmup runs (>= 5 by contract; excluded from measurements)
    warmed = 0
    for w_i in range(max(1, warmups) * len(examples)):
        if warmed >= warmups:
            break
        _, ex_w, gold_w = examples[w_i % len(examples)]
        res_w = _measure_single_request(
            runner, ex_w, gold_w, max_option_len, max_context_tokens
        )
        if not res_w.get("failed"):
            warmed += 1

    query_wall: Dict[str, float] = {}
    query_gpu: Dict[str, float] = {}
    hits: List[float] = []
    peaks: List[float] = []
    gflops_list: List[float] = []
    phi_list: List[float] = []
    tok_list: List[float] = []
    pad_list: List[float] = []
    surv_rows: List[List[bool]] = []

    for qid, ex, gold_id in examples:
        rep_walls: List[float] = []
        rep_gpus: List[float] = []
        last_ok: Optional[Dict[str, Any]] = None
        failed = False
        for _ in range(max(1, repetitions)):
            r = _measure_single_request(
                runner, ex, gold_id, max_option_len, max_context_tokens
            )
            if r.get("failed"):
                raw_queries.append({"qid": qid, "K": K, **r})
                failed = True
                break
            rep_walls.append(float(r["wall_ms"]))
            rep_gpus.append(float(r["gpu_ms"]))
            last_ok = r
        if failed or last_ok is None:
            continue

        # Aggregate repetitions within a query first (AGENTS.md line 220)
        q_wall = sum(rep_walls) / len(rep_walls)
        q_gpu = sum(rep_gpus) / len(rep_gpus)
        query_wall[qid] = q_wall
        query_gpu[qid] = q_gpu
        hits.append(float(last_ok["hit"]))
        peaks.append(float(last_ok["peak_memory_mb"]))
        tok_list.append(float(last_ok["total_tokens"]))
        pad_list.append(float(last_ok["padding_tokens"]))
        surv_rows.append(list(last_ok["gold_survival"]))
        from .operations import profile_request
        module = runner if hasattr(runner,"parameters") else getattr(runner,"backbone",None)
        uses_cuda = module is not None and any(p.is_cuda for p in module.parameters())
        audit = profile_request(lambda: _measure_single_request(runner, ex, gold_id,
            max_option_len, max_context_tokens), cuda=uses_cuda)
        profiles[qid] = audit
        if audit["gpu_kernel_ms"] is not None:
            query_gpu[qid] = audit["gpu_kernel_ms"]
        else:
            query_gpu.pop(qid, None)
        if not audit["failed"]:
            gflops_list.append(audit["gflops"])
        phi_list.append(analytic_request_cost(runner,last_ok))
        raw_queries.append({"qid": qid, "K": K, **last_ok, "repetition_wall_ms": rep_walls,
                            "wall_ms": q_wall, "gpu_kernel_ms": audit["gpu_kernel_ms"]})

    n_ok = len(query_wall)
    n_tot = len(examples)
    wall_vals = list(query_wall.values())
    gpu_vals = list(query_gpu.values())
    stage_surv: List[float] = []
    if surv_rows:
        m_st = len(surv_rows[0])
        stage_surv = [
            sum(1.0 if row[s] else 0.0 for row in surv_rows) / len(surv_rows)
            for s in range(m_st)
        ]

    return Exp5MethodKResult(
        method_name=method_name,
        K=K,
        coverage=n_ok / max(1, n_tot),
        num_successful_queries=n_ok,
        num_total_queries=n_tot,
        hit_at_1=sum(hits) / max(1,n_tot),
        wall_p50_ms=percentile(wall_vals, 50.0) if wall_vals else float("nan"),
        wall_p95_ms=percentile(wall_vals, 95.0) if wall_vals else float("nan"),
        wall_p50_bootstrap=family_bootstrap_ci(
            query_wall,
            n_boot=n_boot,
            agg_fn=lambda xs: percentile(xs, 50.0),
        ),
        gpu_p50_ms=percentile(gpu_vals, 50.0) if gpu_vals else float("nan"),
        gpu_p95_ms=percentile(gpu_vals, 95.0) if gpu_vals else float("nan"),
        peak_memory_mb=max(peaks) if peaks else float("nan"),
        mean_gflops=(sum(gflops_list) / len(gflops_list)) if gflops_list else float("nan"),
        mean_analytic_phi=(sum(phi_list) / len(phi_list)) if phi_list else float("nan"),
        mean_total_tokens=(sum(tok_list) / len(tok_list)) if tok_list else 0.0,
        mean_padding_tokens=(sum(pad_list) / len(pad_list)) if pad_list else 0.0,
        stagewise_gold_survival=stage_surv,
        query_wall_ms=query_wall,
        raw_queries=raw_queries, operation_profiles=profiles,
        wall_p95_bootstrap=family_bootstrap_ci(query_wall,n_boot=n_boot,
                            agg_fn=lambda xs: percentile(xs,95.0)),
    )


def compute_matched_query_speedup(
    baseline_res: Exp5MethodKResult, target_res: Exp5MethodKResult
) -> Optional[float]:
    """Compute speedup (baseline p50 / target p50) ONLY on shared successful queries.

    Never returns infinity for a failed baseline (AGENTS.md line 223).
    """
    shared_qids = sorted(
        set(baseline_res.query_wall_ms.keys()) & set(target_res.query_wall_ms.keys())
    )
    if not shared_qids:
        return None
    b_p50 = percentile([baseline_res.query_wall_ms[q] for q in shared_qids], 50.0)
    t_p50 = percentile([target_res.query_wall_ms[q] for q in shared_qids], 50.0)
    if t_p50 <= 0:
        return None
    return b_p50 / t_p50


def run_exp5_efficiency_panel(
    size_label: str,
    methods: Dict[str, Union[DiffJevModel, BaselineAdapter]],
    specs: Sequence[ScalingQuerySpec],
    ks: Sequence[int] = DEFAULT_SCALING_KS,
    num_queries: int = 100,
    warmups: int = 5,
    repetitions: int = 3,
    max_option_len: Optional[int] = None,
    max_context_tokens: int = 262144,
    n_boot: int = 200,
) -> Dict[str, Any]:
    """Run Exp 5 across ``ks`` for all methods at a given backbone size."""
    gc.collect()
    by_method_k: Dict[str, Dict[int, Exp5MethodKResult]] = {m: {} for m in methods}
    param_info: Dict[str, Dict[str, int]] = {}

    for mname, source in methods.items():
        runner = source() if callable(source) and not hasattr(source, "predict") else source
        bb = getattr(runner, "backbone", None)
        if bb is not None:
            param_info[mname] = parameter_counts(bb)
        for K in ks:
            by_method_k[mname][K] = time_method_at_k(
                method_name=mname,
                runner=runner,
                specs=specs,
                K=K,
                num_queries=num_queries,
                warmups=warmups,
                repetitions=repetitions,
                max_option_len=max_option_len,
                max_context_tokens=max_context_tokens,
                n_boot=n_boot,
            )

        if callable(source) and not hasattr(source, "predict"):
            del bb, runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Build compact comparison table with matched-query speedups
    rows: List[Dict[str, Any]] = []
    for K in ks:
        ar_res = by_method_k.get("serialized_ar", {}).get(K)
        dlm_res = by_method_k.get("serialized_dlm", {}).get(K)
        for mname in methods:
            r = by_method_k[mname][K]
            rows.append(
                {
                    "size": size_label,
                    "method": mname,
                    "K": K,
                    "coverage": r.coverage,
                    "hit_at_1": r.hit_at_1,
                    "stagewise_gold_survival": r.stagewise_gold_survival,
                    "wall_p50_ms": r.wall_p50_ms,
                    "wall_p95_ms": r.wall_p95_ms,
                    "wall_p50_ci": (
                        r.wall_p50_bootstrap["ci_low"],
                        r.wall_p50_bootstrap["ci_high"],
                    ),
                    "gpu_p50_ms": r.gpu_p50_ms,
                    "gpu_p95_ms": r.gpu_p95_ms,
                    "wall_p95_ci": r.wall_p95_bootstrap,
                    "peak_memory_mb": r.peak_memory_mb,
                    "mean_gflops": r.mean_gflops,
                    "flops_scope": "executed shapes with causal fused-SDPA logical pairs; dense bound and unclassified calls retained for kernel audit",
                    "parameter_counts": param_info.get(mname),
                    "speedup_vs_ar": (
                        compute_matched_query_speedup(ar_res, r) if ar_res else None
                    ),
                    "speedup_vs_serialized_dlm": (
                        compute_matched_query_speedup(dlm_res, r) if dlm_res else None
                    ),
                }
            )

    return {
        "size": size_label,
        "parameter_counts": param_info,
        "table": rows,
        "raw": by_method_k,
    }


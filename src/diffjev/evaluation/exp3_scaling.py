"""Exp 3 — Long-tail candidate-count scaling (AGENTS.md section 6, Exp 3).

Protocol:
- Uses ToolRet as a query-only, retrieval-derived single-positive tool-routing
  task (512 stratified evaluation queries, or 32/64 for smoke runs).
- Evaluates ``K in {4, 16, 64, 256, 1024, 2048}`` (cardinality extrapolation
  beyond the ``K <= 32`` training cap for ``K > 32``).
- Constructs one **nested random-negative pool** and one **nested BM25-hard
  negative pool** per query, retaining the annotated positive, excluding known
  positives, and deduplicating tool descriptions/IDs.
- All models receive identical candidate sets and shared truncation rules.
- Reports Hit@1, stagewise gold survival (after pass 1, pass 2, pass 3), and
  context-limit / OOM failures by ``K``.
- Supports optional lower-layer feature cache reuse across ``K`` for accuracy
  sweeps, explicitly separated from Exp 5's fresh full-request timing.
- Exports the frozen BM25-hard negative manifests for reuse in Exp 5.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..baselines.base import BaselineAdapter
from ..baselines.retrieval import BM25Retriever
from ..encoder import BoundaryState, encode_branches
from ..model import DiffJevModel
from ..packing import Candidate, PackedInput, pack_branches
from ..tasks import REFERENCE_ID, DecisionExample
from .metrics import family_bootstrap_ci, hit_at_1

DEFAULT_SCALING_KS: Tuple[int, ...] = (4, 16, 64, 256, 1024, 2048)


@dataclass
class ScalingQuerySpec:
    """One single-positive tool-routing query and its nested negative pools."""

    qid: str
    query_ids: List[int]
    positive: Candidate
    random_negatives: List[Candidate]  # nested: take first K - 1
    hard_negatives: List[Candidate]  # nested BM25-ranked: take first K - 1
    domain: str = "toolret"
    query_text: Optional[str] = None

    def build_example(
        self, K: int, pool: str = "hard", seed: int = 0, include_positive: bool = True
    ) -> DecisionExample:
        if pool not in ("random", "hard"):
            raise ValueError(f"pool must be 'random' or 'hard', got {pool!r}")
        negs = self.hard_negatives if pool == "hard" else self.random_negatives
        needed = (K - 1) if include_positive else K
        if len(negs) < needed:
            raise ValueError(f"{self.qid}: requested K={K}, only {len(negs)} unique negatives available")
        selected_negs = list(negs[:needed])
        cands = ([self.positive] + selected_negs) if include_positive else selected_negs
        # Deterministically shuffle candidate position so positive is not fixed at index 0
        rng = random.Random(f"{seed}:{self.qid}:{K}:{pool}:{include_positive}")
        order = list(range(len(cands)))
        rng.shuffle(order)
        cands = [cands[i] for i in order]
        target = torch.zeros(len(cands) + 1, dtype=torch.float64)
        if include_positive:
            pos_idx = [c.cid for c in cands].index(self.positive.cid)
            target[1 + pos_idx] = 1.0
        else:
            target[0] = 1.0
        return DecisionExample(
            mode="choice",
            query_ids=list(self.query_ids),
            candidates=cands,
            target=target,
            forced_choice=include_positive,
            meta={"qid": self.qid, "K": len(cands), "pool": pool,
                  **({"query_text": self.query_text} if self.query_text is not None else {})},
        )


def build_nested_scaling_manifests(
    queries: Sequence[Dict[str, Any]],
    candidate_corpus: Sequence[Candidate],
    max_k: int = 2048,
    num_queries: int = 512,
    seed: int = 0,
    bm25: Optional[BM25Retriever] = None,
) -> List[ScalingQuerySpec]:
    """Construct nested random-negative and BM25-hard negative pools per query.

    Excludes any known positives for the query and deduplicates candidate IDs
    and token sequences.
    """
    bm25 = bm25 or BM25Retriever()
    rng = random.Random(seed)

    # Deduplicate global corpus by cid and token sequence
    dedup_corpus: List[Candidate] = []
    seen_cids = set()
    seen_toks = set()
    for c in candidate_corpus:
        t_key = c.text if c.text is not None else tuple(c.token_ids)
        if c.cid in seen_cids or t_key in seen_toks:
            continue
        seen_cids.add(c.cid)
        seen_toks.add(t_key)
        dedup_corpus.append(c)

    by_cid = {c.cid: c for c in dedup_corpus}
    from .sampling import stratified_subset
    selected_queries = stratified_subset(queries, num_queries, seed,
        stratum=lambda q: q.get("domain", "toolret"), identity=lambda q: q["qid"])
    specs: List[ScalingQuerySpec] = []

    for q in selected_queries:
        qid = str(q["qid"])
        q_ids = [int(t) for t in q["query_ids"]]
        pos: Candidate = q["positive"]
        known_pos_ids = set(q.get("known_positive_ids") or [pos.cid])
        pos_tok_key = pos.text if pos.text is not None else tuple(pos.token_ids)

        eligible = [
            c
            for c in dedup_corpus
            if c.cid not in known_pos_ids and (c.text if c.text is not None else tuple(c.token_ids)) != pos_tok_key
        ]
        if len(eligible) < max_k - 1:
            raise ValueError(f"{qid}: insufficient unique negatives for K={max_k}")
        need = min(len(eligible), max_k)

        # 1) Nested random pool
        rand_pool = list(eligible)
        rng.shuffle(rand_pool)
        rand_negs = rand_pool[:need]

        # 2) Nested BM25-hard pool
        ranked_ids = bm25.rank_hard_negatives(
            q.get("query_text", "").lower().split() if q.get("query_text") else q_ids,
            [(c.cid, c.text.lower().split() if c.text is not None else c.token_ids) for c in eligible]
        )
        hard_negs = [by_cid[cid] for cid in ranked_ids[:need]]

        specs.append(
            ScalingQuerySpec(
                qid=qid,
                query_ids=q_ids,
                positive=pos,
                random_negatives=rand_negs,
                hard_negatives=hard_negs,
                domain=str(q.get("domain", "toolret")),
                query_text=q.get("query_text"),
            )
        )
    return specs


def save_scaling_manifest(specs: Sequence[ScalingQuerySpec], path: str) -> None:
    """Persist all content needed to replay pools without the original corpus."""
    from dataclasses import asdict
    from pathlib import Path
    from ..data.native import digest
    catalog = {}
    for s in specs:
        for candidate in [s.positive] + list(s.random_negatives) + list(s.hard_negatives):
            record = asdict(candidate)
            if candidate.cid in catalog and catalog[candidate.cid] != record:
                raise ValueError(f"Conflicting content for candidate ID {candidate.cid}")
            catalog[candidate.cid] = record
    queries = [
        {
            "qid": s.qid,
            "domain": s.domain,
            "query_ids": s.query_ids,
            "query_text": s.query_text,
            "positive_id": s.positive.cid,
            "random_negative_ids": [c.cid for c in s.random_negatives],
            "hard_negative_ids": [c.cid for c in s.hard_negatives],
        }
        for s in specs
    ]
    payload = {"version": 2, "candidates": catalog, "queries": queries}
    payload["sha256"] = digest(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_scaling_manifest(path: str) -> List[ScalingQuerySpec]:
    from ..data.native import digest
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload,dict) or payload.get("version") != 2:
        raise ValueError("Legacy ID-only manifest cannot be replayed without its original corpus")
    checksum = payload.pop("sha256")
    if digest(payload) != checksum:
        raise ValueError("Scaling manifest checksum mismatch")
    candidates = {cid: Candidate(**row) for cid,row in payload["candidates"].items()}
    return [ScalingQuerySpec(q["qid"],q["query_ids"],candidates[q["positive_id"]],
                [candidates[cid] for cid in q["random_negative_ids"]],
                [candidates[cid] for cid in q["hard_negative_ids"]],q["domain"],q["query_text"])
            for q in payload["queries"]]


@dataclass
class Exp3ScalingReport:
    model_name: str
    pool: str
    warm_lower_cache: bool
    by_k: Dict[int, Dict[str, Any]]
    no_answer_probe: Optional[Dict[int, Dict[str, float]]] = None
    raw_predictions: List[Dict[str, Any]] = field(default_factory=list)


@torch.inference_mode()
def run_exp3_scaling(
    model_name: str,
    runner: Union[DiffJevModel, BaselineAdapter],
    specs: Sequence[ScalingQuerySpec],
    ks: Sequence[int] = DEFAULT_SCALING_KS,
    pool: str = "hard",
    max_option_len: Optional[int] = None,
    warm_lower_cache: bool = False,
    run_no_answer_probe: bool = False,
    n_boot: int = 200,
) -> Exp3ScalingReport:
    """Evaluate candidate-count scaling across ``ks`` on ``specs``.

    When ``warm_lower_cache=True`` and ``runner`` is a ``DiffJevModel``, each
    query's largest candidate set is encoded once through the lower layers and
    sliced for smaller ``K`` (disclosed separately via ``warm_lower_cache``;
    never used in Exp 5 fresh-request timing).
    """
    ks_sorted = sorted(int(k) for k in ks)
    by_k_preds: Dict[int, List[Optional[str]]] = {k: [] for k in ks_sorted}
    by_k_golds: Dict[int, List[str]] = {k: [] for k in ks_sorted}
    by_k_fam_hits: Dict[int, Dict[str, float]] = {k: {} for k in ks_sorted}
    by_k_survival: Dict[int, List[List[bool]]] = {k: [] for k in ks_sorted}
    by_k_failures: Dict[int, int] = {k: 0 for k in ks_sorted}
    raw_predictions = []
    if hasattr(runner, "eval"):
        runner.eval()

    for spec in specs:
        cached_boundary_by_cid: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        cached_query_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if warm_lower_cache and isinstance(runner, DiffJevModel):
            ex_max = spec.build_example(max(ks_sorted), pool=pool)
            packed_max = runner.pack(ex_max, max_option_len=max_option_len)
            bd_max = runner.encode(packed_max)
            cached_query_state = (bd_max.query_h, bd_max.query_pos)
            for i, cid in enumerate(packed_max.candidate_ids):
                cached_boundary_by_cid[cid] = (
                    bd_max.h_slot_base[i],
                    bd_max.option_h[i],
                    bd_max.option_pos[i],
                    bd_max.option_valid[i],
                    bd_max.slot_pos[i],
                )

        for K in ks_sorted:
            ex = spec.build_example(K, pool=pool)
            from ..preprocessing import retokenize_example, runner_tokenizer
            ex = retokenize_example(ex, runner_tokenizer(runner), max_option_len)
            gold_id = spec.positive.cid
            by_k_golds[K].append(gold_id)

            if isinstance(runner, DiffJevModel):
                try:
                    packed = runner.pack(ex, max_option_len=max_option_len)
                    if warm_lower_cache and cached_query_state is not None:
                        cids = packed.candidate_ids
                        bd = BoundaryState(
                            h_slot_base=torch.stack([cached_boundary_by_cid[c][0] for c in cids]),
                            query_h=cached_query_state[0],
                            query_pos=cached_query_state[1],
                            option_h=torch.stack([cached_boundary_by_cid[c][1] for c in cids]),
                            option_pos=torch.stack([cached_boundary_by_cid[c][2] for c in cids]),
                            option_valid=torch.stack([cached_boundary_by_cid[c][3] for c in cids]),
                            slot_pos=torch.stack([cached_boundary_by_cid[c][4] for c in cids]),
                        )
                        ro = runner.denoiser.rollout(bd, ex.mode, full_support=False)
                        out = runner.readout(ex, packed, ro, gold_id=gold_id)
                    else:
                        bd = runner.encode(packed)
                        ro = runner.denoiser.rollout(bd, ex.mode, full_support=False)
                        out = runner.readout(ex, packed, ro, gold_id=gold_id)
                except (RuntimeError, ValueError) as e:
                    if "out of memory" in str(e).lower() or "context_limit" in str(e):
                        by_k_failures[K] += 1
                        by_k_preds[K].append(None)
                        by_k_fam_hits[K][spec.qid] = 0.0
                        raw_predictions.append({"qid":spec.qid,"K":K,"failed":True,"reason":"context_limit" if "context_limit" in str(e) else "oom"})
                        continue
                    raise
                by_k_preds[K].append(out.top1)
                by_k_fam_hits[K][spec.qid] = 1.0 if out.top1 == gold_id else 0.0
                # Record stagewise gold survival after pass 1, pass 2, pass 3
                final_surv = gold_id in set(out.candidate_ids)
                stage_surv = [bool(s) for s in out.gold_survival]
                by_k_survival[K].append(stage_surv)
                raw_predictions.append({"qid":spec.qid,"K":len(ex.candidates),"failed":False,**out.to_record()})
            else:
                res = runner.predict(ex)
                raw_predictions.append({"qid":spec.qid,"K":len(ex.candidates),"top1":res.top1,
                    "failed":res.failed,"failure_reason":res.failure_reason,"probabilities":res.prob_by_id})
                if res.failed:
                    by_k_failures[K] += 1
                    by_k_preds[K].append(None)
                    by_k_fam_hits[K][spec.qid] = 0.0
                else:
                    by_k_preds[K].append(res.top1)
                    by_k_fam_hits[K][spec.qid] = 1.0 if res.top1 == gold_id else 0.0

    by_k_summary: Dict[int, Dict[str, Any]] = {}
    for K in ks_sorted:
        surv_lists = by_k_survival[K]
        stage_rates: List[float] = []
        if surv_lists:
            n_stages = len(surv_lists[0])
            stage_rates = [
                sum(1.0 if row[s] else 0.0 for row in surv_lists) / len(surv_lists)
                for s in range(n_stages)
            ]
        by_k_summary[K] = {
            "K": K,
            "cardinality_regime": "in_distribution" if K <= 32 else "extrapolation_beyond_train_cap",
            "num_queries": len(specs),
            "failures": by_k_failures[K],
            "coverage": (len(specs) - by_k_failures[K]) / max(1, len(specs)),
            "hit_at_1": hit_at_1(by_k_preds[K], by_k_golds[K]),
            "bootstrap_hit_at_1": family_bootstrap_ci(by_k_fam_hits[K], n_boot=n_boot),
            "stagewise_gold_survival": stage_rates,
        }

    probe_summary: Optional[Dict[int, Dict[str, float]]] = None
    if run_no_answer_probe and isinstance(runner, DiffJevModel):
        probe_summary = {}
        for K in ks_sorted:
            abstains = 0
            ref_probs: List[float] = []
            for spec in specs:
                ex_null = spec.build_example(K, pool=pool, include_positive=False)
                out_null = runner.predict(ex_null, full_support=False)
                if out_null.top1 == REFERENCE_ID:
                    abstains += 1
                ref_probs.append(out_null.reference_prob)
            probe_summary[K] = {
                "abstention_rate": abstains / max(1, len(specs)),
                "mean_reference_prob": sum(ref_probs) / max(1, len(ref_probs)),
            }

    return Exp3ScalingReport(
        model_name=model_name,
        pool=pool,
        warm_lower_cache=warm_lower_cache,
        by_k=by_k_summary,
        no_answer_probe=probe_summary,
        raw_predictions=raw_predictions,
    )


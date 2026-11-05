"""Freezing policy, parameter accounting, checkpointing, and training pipelines.

Policy (AGENTS.md sections 2 and 5):
- Train only the final ``k = 4`` existing transformer layers and the final
  normalization; freeze lower layers, input embeddings, and the output head.
- Add no new trainable neural layers, embeddings, adapters, or LoRA modules.
- Default training budget: effective batch size 32 (8 per bucket x 4 buckets),
  3,125 optimizer updates = 100,000 example presentations, AdamW ``lr = 2e-5``,
  bf16 where supported, gradient clipping at 1.0.
- Also provides ``train_synthetic`` for quick unit tests and mock smoke runs.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import hashlib
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .backbones.base import BackboneAdapter
from .config import TrainConfig
from .data.sources import materialize_example
from .model import DiffJevModel
from .tasks import DecisionExample


def unique_parameters(module: nn.Module) -> List[nn.Parameter]:
    seen, out = set(), []
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def apply_freeze_policy(backbone: BackboneAdapter, num_top_layers: int = 4) -> Dict[str, int]:
    for p in backbone.parameters():
        p.requires_grad_(False)
    layers = backbone.layer_modules()
    for i in backbone.top_layer_indices(num_top_layers):
        for p in layers[i].parameters():
            p.requires_grad_(True)
    for p in backbone.final_norm_module().parameters():
        p.requires_grad_(True)
    # Embeddings / head stay frozen even if tied to a trainable module elsewhere.
    for m in (backbone.input_embedding_module(), backbone.output_head_module()):
        for p in m.parameters():
            p.requires_grad_(False)
    return parameter_counts(backbone)


def parameter_counts(module: nn.Module) -> Dict[str, int]:
    ps = unique_parameters(module)
    return {
        "total": int(sum(p.numel() for p in ps)),
        "trainable": int(sum(p.numel() for p in ps if p.requires_grad)),
    }


def trainable_parameter_names(module: nn.Module) -> List[str]:
    return [n for n, p in module.named_parameters() if p.requires_grad]


def save_trainable_checkpoint(
    path: str,
    model: DiffJevModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Save only the trainable parameter tensors, config, and optimizer state."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    trainable_names = set(trainable_parameter_names(model.backbone))
    state = {
        k: v.detach().cpu()
        for k, v in model.backbone.state_dict().items()
        if k in trainable_names
    }
    payload: Dict[str, Any] = {
        "step": int(step),
        "diffjev_config": model.cfg.to_dict(),
        "trainable_state_dict": state,
        "parameter_counts": parameter_counts(model.backbone),
        "extra_meta": extra_meta or {},
        "method": type(model).__name__,
        "backbone_provenance": getattr(model.backbone, "provenance", {}),
        "runtime_state": getattr(model, "runtime_state", {}),
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_trainable_checkpoint(
    path: str,
    model: DiffJevModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    """Load trainable weights (and optional optimizer state) into ``model``."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("method") != type(model).__name__:
        raise ValueError("Checkpoint method mismatch: baselines must be adapted separately")
    if payload.get("backbone_provenance", {}) != getattr(model.backbone, "provenance", {}):
        raise ValueError("Checkpoint backbone revision mismatch")
    expected = set(trainable_parameter_names(model.backbone))
    if expected and set(payload["trainable_state_dict"]) != expected:
        raise ValueError("Checkpoint trainable parameter set does not match the frozen-layer policy")
    missing, unexpected = model.backbone.load_state_dict(
        payload["trainable_state_dict"], strict=False
    )
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if "diffjev_config" in payload and "readout_temperature" in payload["diffjev_config"]:
        saved = dict(payload["diffjev_config"])
        current = model.cfg.to_dict()
        saved.pop("readout_temperature",None);current.pop("readout_temperature",None)
        if saved != current:
            raise ValueError("Checkpoint architecture/objective configuration mismatch")
        model.cfg.readout_temperature.update(
            payload["diffjev_config"]["readout_temperature"]
        )
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def capture_rng_state():
    import numpy as np
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    import numpy as np
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate_validation(
    model: DiffJevModel,
    examples: Sequence[DecisionExample],
    seed: int = 0,
) -> Dict[str, float]:
    """Compute validation losses and top-1 accuracy on a development split."""
    if not examples:
        return {"val_total": float("nan"), "val_task_ce": float("nan"), "val_acc": float("nan")}
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    totals: Dict[str, float] = Counter()
    correct = 0
    scored = 0
    for ex in examples:
        losses = model.training_losses(ex, gen)
        for k, v in losses.items():
            totals[k] += float(v.item() if hasattr(v, "item") else v)
        out = model.predict(ex, full_support=True) if isinstance(model, DiffJevModel) else model.predict(ex)
        pm = out.prob_by_id() if isinstance(model, DiffJevModel) else out.prob_by_id
        if pm is None or getattr(out, "failed", False):
            raise ValueError("Validation needs successful full-support probabilities")
        keys = ["__reference__"] + [c.cid for c in ex.candidates]
        totals["decision_ce"] += sum(-float(t)*math.log(max(pm.get(cid, 0), 1e-30))
            for cid,t in zip(keys, ex.target) if t > 0)
        if ex.gold_id is not None:
            scored += 1
            if out.top1 == ex.gold_id:
                correct += 1
    if was_training:
        model.train()
    n = len(examples)
    return {
        "val_total": totals["total"] / n,
        "val_task_ce": totals.get("L_task_CE", 0.0) / n,
        "val_mask": totals.get("L_mask", 0.0) / n,
        "val_brier": totals.get("L_Brier", 0.0) / n,
        "val_acc": (correct / scored) if scored else float("nan"),
        "val_decision_ce": totals["decision_ce"] / n,
    }


def train_synthetic(
    model: DiffJevModel,
    examples: Sequence[DecisionExample],
    num_updates: int,
    effective_batch: int = 8,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    seed: int = 0,
    log_fn: Optional[Callable[[Dict], None]] = None,
) -> List[Dict[str, float]]:
    """Smoke-training loop on synthetic examples (sampling with replacement)."""
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in unique_parameters(model) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    history: List[Dict[str, float]] = []
    model.train()
    for step in range(num_updates):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        agg: Dict[str, float] = {}
        idx = torch.randint(len(examples), (effective_batch,), generator=gen).tolist()
        for i in idx:
            losses = model.training_losses(examples[i], gen)
            (losses["total"] / effective_batch).backward()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v) / effective_batch
        gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if not math.isfinite(float(gnorm)):
            raise FloatingPointError(f"non-finite grad norm at step {step}")
        opt.step()
        agg["grad_norm"] = float(gnorm)
        agg["step"] = step
        agg["sec"] = time.time() - t0
        history.append(agg)
        if log_fn is not None:
            log_fn(agg)
    return history


def train_diffjev(
    model: DiffJevModel,
    train_cfg: TrainConfig,
    records_by_id: Dict[str, Dict[str, Any]],
    trace: Sequence[Sequence[Dict[str, str]]],
    tokenizer,
    banks_by_domain: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    dev_examples: Optional[Sequence[DecisionExample]] = None,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    save_every: int = 500,
    eval_every: int = 250,
    enable_augmentation: bool = True,
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Formal supervised structured-denoising training loop (AGENTS.md section 5).

    Replays the exact batch trace across ``len(trace)`` optimizer updates,
    accumulating gradients across each effective batch (default 32 examples =
    8 per bucket x 4 buckets), and logs unique record counts, repetition rates,
    processed tokens, realized candidate-count histogram, and estimated FLOPs.
    """
    counts = apply_freeze_policy(model.backbone, model.cfg.num_top_layers)
    params = [p for p in unique_parameters(model) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=train_cfg.learning_rate)

    start_step = 0
    ckpt = None
    if resume_from and not os.path.isfile(resume_from):
        raise FileNotFoundError(resume_from)
    if resume_from and os.path.isfile(resume_from):
        ckpt = load_trainable_checkpoint(resume_from, model, optimizer=opt)
        start_step = int(ckpt.get("step", 0)) + 1

    device = model.backbone.input_embedding_module().weight.device
    use_bf16 = bool(
        train_cfg.bf16
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    gen = torch.Generator().manual_seed(train_cfg.seed)
    aug_rng = random.Random(train_cfg.seed + 10000)

    history: List[Dict[str, Any]] = []
    seen_rids: Counter = Counter()
    realized_k_hist: Counter = Counter()
    null_aug_count = 0
    processed_tokens = 0
    best_val_loss = float("inf")
    training_wall_seconds = 0.0

    trace_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    if ckpt:
        state = ckpt["runtime_state"]
        if state["trace_hash"] != trace_hash or state["train_config"] != train_cfg.to_dict():
            raise ValueError("Resume trace or training configuration changed")
        if state["enable_augmentation"] != enable_augmentation:
            raise ValueError("Resume augmentation policy changed")
        gen.set_state(state["generator"].cpu())
        aug_rng.setstate(state["augmentation_rng"])
        restore_rng_state(ckpt["rng_state"])
        seen_rids = Counter(state["seen_rids"])
        realized_k_hist = Counter(state["realized_k_hist"])
        processed_tokens = state["processed_tokens"]
        null_aug_count = state["null_aug_count"]
        best_val_loss = state["best_val_loss"]
        training_wall_seconds = state.get("training_wall_seconds",0.0)

    def checkpoint_state():
        model.runtime_state = {"trace_hash": trace_hash, "train_config": train_cfg.to_dict(),
            "generator": gen.get_state(), "augmentation_rng": aug_rng.getstate(),
            "seen_rids": dict(seen_rids), "realized_k_hist": dict(realized_k_hist),
            "processed_tokens": processed_tokens, "null_aug_count": null_aug_count,
            "best_val_loss": best_val_loss, "enable_augmentation": enable_augmentation}
        model.runtime_state["training_wall_seconds"] = training_wall_seconds

    model.train()
    for step in range(start_step, len(trace)):
        t0 = time.time()
        batch_spec = trace[step]
        bsz = len(batch_spec)
        opt.zero_grad(set_to_none=True)
        agg: Dict[str, float] = {}

        for item in batch_spec:
            rid = item["rid"]
            seen_rids[rid] += 1
            rec = dict(records_by_id[rid])
            frozen = enable_augmentation and "options" in item
            if frozen:
                rec.update(options=item["options"], target=item["target"], is_retrieval_pool=False)
            dom = rec.get("domain", "general")
            bank = (banks_by_domain or {}).get(dom, [])
            ex, meta = materialize_example(
                tokenizer,
                rec,
                negative_bank=bank,
                rng=aug_rng,
                enable_augmentation=enable_augmentation and not frozen,
                k_proposal=train_cfg.k_proposal,
                null_rate=train_cfg.null_augmentation_rate,
                max_k=train_cfg.max_train_k,
            )
            if frozen:
                meta = item["augmentation"]
            K = len(ex.candidates)
            realized_k_hist[K] += 1
            if meta.get("null_augmented"):
                null_aug_count += 1
            q_len = len(ex.query_ids)
            m_avg = int(round(sum(len(c.token_ids) for c in ex.candidates) / max(1, K)))
            processed_tokens += q_len + sum(1 + len(c.token_ids) for c in ex.candidates)

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = model.training_losses(ex, gen)
                    (losses["total"] / bsz).backward()
            else:
                losses = model.training_losses(ex, gen)
                (losses["total"] / bsz).backward()

            for lk, lv in losses.items():
                agg[lk] = agg.get(lk, 0.0) + float(lv.detach().item()) / bsz

        gnorm = torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
        if not math.isfinite(float(gnorm)):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        opt.step()

        row: Dict[str, Any] = {
            "step": step,
            "sec": time.time() - t0,
            "grad_norm": float(gnorm),
            "processed_tokens": processed_tokens,
            **agg,
        }
        training_wall_seconds += row["sec"]

        if dev_examples and ((step + 1) % eval_every == 0 or step == len(trace) - 1):
            val_metrics = evaluate_validation(model, dev_examples, seed=train_cfg.seed)
            row.update(val_metrics)
            if checkpoint_dir and val_metrics["val_decision_ce"] < best_val_loss:
                best_val_loss = val_metrics["val_decision_ce"]
                checkpoint_state()
                save_trainable_checkpoint(
                    os.path.join(checkpoint_dir, "best.pt"),
                    model,
                    optimizer=opt,
                    step=step,
                    extra_meta={"val": val_metrics},
                )

        checkpoint_state()
        if checkpoint_dir and ((step + 1) % save_every == 0 or step == len(trace) - 1):
            save_trainable_checkpoint(
                os.path.join(checkpoint_dir, f"step_{step + 1}.pt"),
                model,
                optimizer=opt,
                step=step,
            )

        history.append(row)
        if log_fn is not None:
            log_fn(row)

    total_presentations = sum(seen_rids.values())
    unique_records = len(seen_rids)
    summary = {
        "parameter_counts": counts,
        "enabled_loss_terms": model.cfg.enabled_loss_terms(),
        "optimizer_updates": len(trace),
        "updates_this_session": len(trace) - start_step,
        "trace_hash": trace_hash,
        "checkpoint_selection": "development full-support decision CE",
        "total_presentations": total_presentations,
        "unique_records": unique_records,
        "repetition_rate": (total_presentations - unique_records) / max(1, total_presentations),
        "processed_tokens": processed_tokens,
        "realized_k_histogram": dict(sorted(realized_k_hist.items())),
        "null_augmented_presentations": null_aug_count,
        "training_flops": None,
        "training_flops_note": "Requires a separate executed-operator training profile; no proxy factor reported.",
        "training_wall_seconds": training_wall_seconds,
    }
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return history, summary

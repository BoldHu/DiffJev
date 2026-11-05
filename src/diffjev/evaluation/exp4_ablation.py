"""Exp 4 — Controlled one-factor ablations on the LLaDA anchor (AGENTS.md section 6, Exp 4).

Variants (one factor changed at a time, never a factorial grid):
1. ``full``                    : Full DiffJev (T=3, Set-RoPE, cross-slot, pruning, K-augmentation)
2. ``full_context_t1_control`` : Separately trained full-context T=1 control (num_passes=1,
                                 use_mask_loss=False, first_pass_detail_memory=True)
3. ``t1_slot_only``            : Truncated T=1 diagnostic (first pass slot-only)
4. ``no_cross_slot``           : cross_slot_attention=False
5. ``untied_positions``        : position_mode='sequential'
6. ``no_pruning``              : pruning_enabled=False
7. ``no_augmentation``         : trained without candidate-count or null augmentation
8. ``no_mask_loss_t3``         : optional matched-step T=3 control without L_mask

Pairs accuracy, order-flip rate, calibration (Brier/ECE), and large-K Hit@1 with
actual training and inference GFLOPs so compute differences are explicitly reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import DiffJevConfig, TrainConfig
from ..model import DiffJevModel
from ..tasks import DecisionExample
from .exp1_benchmark import JevBenchCase, run_exp1_benchmark
from .exp2_robustness import run_exp2_robustness
from .exp3_scaling import ScalingQuerySpec, run_exp3_scaling


@dataclass
class AblationSpec:
    name: str
    description: str
    method_overrides: Dict[str, Any]
    enable_augmentation: bool = True
    requires_separate_training: bool = True


ABLATION_SPECS: Dict[str, AblationSpec] = {
    "full": AblationSpec(
        name="full",
        description="Full DiffJev (T=3, tied Set-RoPE, cross-slot attention, pruning, augmentation)",
        method_overrides={},
        enable_augmentation=True,
        requires_separate_training=True,
    ),
    "full_context_t1_control": AblationSpec(
        name="full_context_t1_control",
        description="Separately trained full-context T=1 control (detail memory in pass 1, no L_mask)",
        method_overrides={
            "num_passes": 1,
            "use_mask_loss": False,
            "first_pass_detail_memory": True,
        },
        enable_augmentation=True,
        requires_separate_training=True,
    ),
    "t1_slot_only": AblationSpec(
        name="t1_slot_only",
        description="Truncated T=1 diagnostic (compressed slot states only)",
        method_overrides={"num_passes": 1, "first_pass_detail_memory": False},
        enable_augmentation=True,
        requires_separate_training=False,
    ),
    "no_cross_slot": AblationSpec(
        name="no_cross_slot",
        description="No cross-slot attention (slots attend only to self + own detail memory)",
        method_overrides={"cross_slot_attention": False},
        enable_augmentation=True,
        requires_separate_training=True,
    ),
    "untied_positions": AblationSpec(
        name="untied_positions",
        description="Untied sequential positions (breaks permutation equivariance)",
        method_overrides={"position_mode": "sequential"},
        enable_augmentation=True,
        requires_separate_training=True,
    ),
    "no_pruning": AblationSpec(
        name="no_pruning",
        description="No hard pruning (all K slots retained across all 3 passes)",
        method_overrides={"pruning_enabled": False},
        enable_augmentation=True,
        requires_separate_training=False,
    ),
    "no_augmentation": AblationSpec(
        name="no_augmentation",
        description="No candidate-count or null augmentation during training",
        method_overrides={},
        enable_augmentation=False,
        requires_separate_training=True,
    ),
    "no_mask_loss_t3": AblationSpec(
        name="no_mask_loss_t3",
        description="Optional matched-step T=3 control without L_mask (isolates masked objective)",
        method_overrides={"use_mask_loss": False},
        enable_augmentation=True,
        requires_separate_training=True,
    ),
}


def build_ablation_config(
    base_cfg: Optional[DiffJevConfig], variant_name: str
) -> Tuple[DiffJevConfig, AblationSpec]:
    if variant_name not in ABLATION_SPECS:
        raise ValueError(f"Unknown ablation variant {variant_name!r}; valid: {sorted(ABLATION_SPECS)}")
    spec = ABLATION_SPECS[variant_name]
    base_dict = (base_cfg or DiffJevConfig()).to_dict()
    base_dict.update(spec.method_overrides)
    return DiffJevConfig.from_dict(base_dict), spec


def run_exp4_ablation_suite(
    model_builder: Callable[[DiffJevConfig, AblationSpec], Tuple[DiffJevModel, Optional[Dict[str, Any]]]],
    benchmark_cases: Sequence[JevBenchCase],
    robustness_families: Sequence[Tuple[str, DecisionExample]],
    scaling_specs: Sequence[ScalingQuerySpec],
    scaling_ks: Sequence[int] = (4, 64, 256),
    variants: Sequence[str] = (
        "full",
        "full_context_t1_control",
        "no_cross_slot",
        "untied_positions",
        "no_pruning",
        "no_augmentation",
    ),
    base_cfg: Optional[DiffJevConfig] = None,
    n_boot: int = 100,
) -> List[Dict[str, Any]]:
    """Run the controlled Exp 4 ablation table and report accuracy + compute."""
    table: List[Dict[str, Any]] = []
    for vname in variants:
        cfg, spec = build_ablation_config(base_cfg, vname)
        model, train_summary = model_builder(cfg, spec)

        exp1 = run_exp1_benchmark(vname, model, benchmark_cases, n_boot=n_boot)
        exp2 = run_exp2_robustness(
            vname, model, robustness_families, max_families=200, n_random_perms=6, n_boot=n_boot
        )
        exp3 = run_exp3_scaling(vname, model, scaling_specs, ks=scaling_ks, pool="hard", n_boot=n_boot)

        # Estimate inference GFLOPs at largest evaluated K to expose compute differences
        k_max = max(scaling_ks)
        ex_sample = scaling_specs[0].build_example(k_max, pool="hard")
        from .exp5_efficiency import _estimate_method_gflops
        gflops_kmax, _ = _estimate_method_gflops(model, ex_sample, [])

        row = {
            "variant": vname,
            "description": spec.description,
            "enabled_loss_terms": cfg.enabled_loss_terms(),
            "benchmark_accuracy": exp1.aggregate["accuracy"],
            "benchmark_brier": exp1.aggregate["brier"],
            "benchmark_ece": exp1.aggregate["ece"],
            "order_flip_rate": exp2.order_permutation["decision_flip_rate"],
            "order_max_prob_drift": exp2.order_permutation["mean_max_prob_drift"],
            "alias_flip_rate": exp2.alias_renaming["decision_flip_rate"],
            "scaling_hit_at_1": {k: exp3.by_k[k]["hit_at_1"] for k in scaling_ks},
            "infer_gflops_at_kmax": gflops_kmax,
            "training_wall_seconds": (train_summary or {}).get("training_wall_seconds"),
            "operation_profile": model.last_operation_profile,
        }
        table.append(row)
    return table


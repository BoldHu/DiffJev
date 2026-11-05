"""Evaluation metrics and runners for the five core experiments (AGENTS.md section 6):

- ``metrics``          : accuracy, flip rate, drift, JS, Brier, ECE, NLL, family bootstrap CI
- ``exp1_benchmark``   : Exp 1 — JevBench benchmark performance
- ``exp2_robustness``  : Exp 2 — Option-order and external-alias robustness
- ``exp3_scaling``     : Exp 3 — Long-tail candidate-count scaling (K up to 2,048)
- ``exp4_ablation``    : Exp 4 — Single-factor controlled ablations on the LLaDA anchor
- ``exp5_efficiency``  : Exp 5 — Matched-size latency, memory, and GFLOPs complexity
"""

from .exp1_benchmark import Exp1ModelReport, JevBenchCase, run_exp1_benchmark
from .exp2_robustness import (
    ALIAS_SCHEMES,
    Exp2RobustnessReport,
    apply_alias_scheme,
    run_exp2_robustness,
)
from .exp3_scaling import (
    DEFAULT_SCALING_KS,
    Exp3ScalingReport,
    ScalingQuerySpec,
    build_nested_scaling_manifests,
    run_exp3_scaling,
    save_scaling_manifest,
)
from .exp4_ablation import (
    ABLATION_SPECS,
    AblationSpec,
    build_ablation_config,
    run_exp4_ablation_suite,
)
from .exp5_efficiency import (
    Exp5MethodKResult,
    compute_matched_query_speedup,
    run_exp5_efficiency_panel,
    time_method_at_k,
)
from .metrics import (
    brier_score,
    chance_corrected_accuracy,
    decision_flip_rate,
    expected_calibration_error,
    family_bootstrap_ci,
    hit_at_1,
    js_divergence,
    max_probability_drift,
    negative_log_likelihood,
    percentile,
    permutations_for,
    score_task_metrics,
)

__all__ = [
    "Exp1ModelReport",
    "JevBenchCase",
    "run_exp1_benchmark",
    "ALIAS_SCHEMES",
    "Exp2RobustnessReport",
    "apply_alias_scheme",
    "run_exp2_robustness",
    "DEFAULT_SCALING_KS",
    "Exp3ScalingReport",
    "ScalingQuerySpec",
    "build_nested_scaling_manifests",
    "run_exp3_scaling",
    "save_scaling_manifest",
    "ABLATION_SPECS",
    "AblationSpec",
    "build_ablation_config",
    "run_exp4_ablation_suite",
    "Exp5MethodKResult",
    "compute_matched_query_speedup",
    "run_exp5_efficiency_panel",
    "time_method_at_k",
    "brier_score",
    "chance_corrected_accuracy",
    "decision_flip_rate",
    "expected_calibration_error",
    "family_bootstrap_ci",
    "hit_at_1",
    "js_divergence",
    "max_probability_drift",
    "negative_log_likelihood",
    "percentile",
    "permutations_for",
    "score_task_metrics",
]

"""Method configuration for DiffJev.

Defaults follow AGENTS.md (sections 2-5). Ablation switches exist only for the
five Exp 4 factors (T = 1, no cross-slot attention, untied sequential
positions, no pruning, no candidate augmentation) and the full-context T = 1
control. They are not additional research tasks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class DiffJevConfig:
    # ---- architecture -------------------------------------------------
    num_top_layers: int = 4
    """k: final transformer layers reused as the slot co-denoiser (trained)."""

    position_mode: str = "tied"
    """'tied' (Set-RoPE, default) or 'sequential' (Exp 4 ablation: untied)."""

    cross_slot_attention: bool = True
    """Exp 4 ablation switch: False lets each slot attend only to itself."""

    branch_chunk_size: int = 64
    """Number of branches encoded together in the lower layers (exact chunking)."""

    # ---- inference rollout -------------------------------------------
    num_passes: int = 3
    """T: denoiser evaluations at t = 1, (T-1)/T, ..., 1/T."""

    first_pass_detail_memory: bool = False
    """False (default): pass 1 sees compressed slot states only.
    True: full-context T = 1 control (Exp 4) reads its own detail memory."""

    # ---- pruning ------------------------------------------------------
    pruning_enabled: bool = True
    prune_min_k: int = 32
    """Hard pruning disabled whenever K <= prune_min_k."""
    prune_schedule: Tuple[int, ...] = (32, 8)
    """Nominal survivor caps before passes 2, 3, ... (ties retained)."""

    # ---- training objective -----------------------------------------
    use_mask_loss: bool = True
    """L_mask. False for the full-context T = 1 control."""
    use_task_ce: bool = True
    brier_weight: float = 0.1
    """0.0 disables Brier explicitly."""
    score_cumulative_brier: bool = False
    """For score mode, optionally replace categorical Brier by cumulative Brier."""
    iia_weight: float = 0.0
    """IIA regulariser defaults to zero (not implemented beyond weight 0)."""
    full_mask_prob: float = 0.5
    """Probability of t = 1 corruption draws; else t ~ U[t_min, 1)."""
    t_min: float = 1e-3
    feedback_stop_gradient: bool = True
    """Stop-gradient through feedback probabilities (recorded choice)."""

    # ---- readout ------------------------------------------------------
    readout_temperature: Dict[str, float] = field(
        default_factory=lambda: {"choice": 1.0, "noul": 1.0, "score": 1.0}
    )
    """Fixed final-readout temperature per mode (fit on calibration split only)."""

    def __post_init__(self) -> None:
        if self.position_mode not in ("tied", "sequential"):
            raise ValueError(f"unknown position_mode {self.position_mode!r}")
        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")
        if self.num_top_layers < 1:
            raise ValueError("num_top_layers must be >= 1")
        if self.iia_weight != 0.0:
            raise NotImplementedError(
                "IIA regularisation defaults to zero and is not implemented in Phase 1."
            )
        self.prune_schedule = tuple(int(x) for x in self.prune_schedule)

    def t_schedule(self) -> Tuple[float, ...]:
        """t values of the T denoiser evaluations: 1, (T-1)/T, ..., 1/T."""
        T = self.num_passes
        return tuple((T - p) / T for p in range(T))

    def enabled_loss_terms(self) -> Dict[str, Any]:
        return {
            "L_mask": self.use_mask_loss,
            "L_task_CE(mean over passes)": self.use_task_ce,
            "L_Brier": self.brier_weight if self.brier_weight > 0 else False,
            "L_IIA": False,
            "feedback_stop_gradient": self.feedback_stop_gradient,
        }

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DiffJevConfig":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown DiffJevConfig keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class TrainConfig:
    """Training budget defaults (AGENTS.md section 5.1)."""

    effective_batch_size: int = 32
    optimizer_updates: int = 3125
    learning_rate: float = 2e-5
    grad_clip: float = 1.0
    bf16: bool = True
    seed: int = 0
    bucket_shares: Dict[str, float] = field(
        default_factory=lambda: {
            "openjev_choice": 0.25,
            "openjev_noul": 0.25,
            "openjev_score": 0.25,
            "toolret_choice": 0.25,
        }
    )
    max_train_k: int = 32
    k_proposal: Dict[int, float] = field(
        default_factory=lambda: {2: 0.50, 4: 0.25, 8: 0.15, 16: 0.07, 32: 0.03}
    )
    null_augmentation_rate: float = 0.15

    @property
    def example_presentations(self) -> int:
        return self.effective_batch_size * self.optimizer_updates

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file (requires PyYAML)."""
    import yaml  # local import: optional dependency

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


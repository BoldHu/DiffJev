"""Standardized baseline interface and result container (AGENTS.md section 7)."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..tasks import DecisionExample


@dataclass
class BaselineResult:
    baseline_name: str
    mode: str
    candidate_ids: List[str]
    top1: Optional[str]
    prob_by_id: Optional[Dict[str, float]] = None
    expected_score: Optional[float] = None
    p_yes: Optional[float] = None
    local_gpu_ms: Optional[float] = None
    api_latency_ms: Optional[float] = None
    total_tokens: int = 0
    failed: bool = False
    failure_reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def probabilities_available(self) -> bool:
        return self.prob_by_id is not None and not self.failed


class BaselineAdapter(abc.ABC):
    name: str = "baseline"

    @abc.abstractmethod
    def predict(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        """Full evaluation prediction (with probabilities if supported)."""

    def predict_top1(
        self, example: DecisionExample, max_context_tokens: Optional[int] = None
    ) -> BaselineResult:
        """Fast top-1 decision used in Exp 5 timing (defaults to ``predict``)."""
        return self.predict(example, max_context_tokens=max_context_tokens)


"""Evaluation metrics and canonical-family bootstrap uncertainty (AGENTS.md sections 6 & 9).

All metrics operate on outputs already remapped to immutable semantic candidate IDs.
Evaluation uncertainty is grouped by canonical query/family, never by reordered copies.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def permutations_for(K: int, n_random: int = 10, seed: int = 0) -> List[Tuple[int, ...]]:
    """Exp 2 protocol: all permutations for K <= 4; else original order plus at most
    ``n_random`` seeded random permutations."""
    if K <= 4:
        return list(itertools.permutations(range(K)))
    rng = random.Random(seed)
    out = [tuple(range(K))]
    seen = set(out)
    attempts = 0
    while len(out) < 1 + n_random and attempts < 100 * n_random:
        p = list(range(K))
        rng.shuffle(p)
        tp = tuple(p)
        attempts += 1
        if tp not in seen:
            seen.add(tp)
            out.append(tp)
    return out


def decision_flip_rate(decisions_per_family: Sequence[Sequence[str]]) -> float:
    """Fraction of canonical families whose semantic decision differs across conditions."""
    if not decisions_per_family:
        return float("nan")
    flips = sum(1 for d in decisions_per_family if len(set(d)) > 1)
    return flips / len(decisions_per_family)


def max_probability_drift(prob_maps: Sequence[Dict[str, float]]) -> float:
    """Maximum absolute probability difference over candidate IDs across orderings."""
    if not prob_maps:
        return 0.0
    keys = set().union(*[set(m) for m in prob_maps])
    drift = 0.0
    for k in keys:
        vals = [m.get(k, 0.0) for m in prob_maps]
        drift = max(drift, max(vals) - min(vals))
    return drift


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a, b):
        return sum(
            a.get(k, 0.0) * math.log(a.get(k, 0.0) / max(b[k], 1e-15))
            for k in keys
            if a.get(k, 0.0) > 0
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hit_at_1(predictions: Sequence[Optional[str]], golds: Sequence[str]) -> float:
    if not golds:
        return float("nan")
    return sum(p == g for p, g in zip(predictions, golds)) / len(golds)


def chance_corrected_accuracy(
    predictions: Sequence[Optional[str]], golds: Sequence[str], num_choices: Sequence[int]
) -> float:
    """Chance-corrected accuracy: ``mean((1[pred == gold] - 1/K) / (1 - 1/K))``."""
    if not golds:
        return float("nan")
    tot = 0.0
    for p, g, k in zip(predictions, golds, num_choices):
        chance = 1.0 / max(2, int(k))
        hit = 1.0 if p == g else 0.0
        tot += (hit - chance) / (1.0 - chance)
    return tot / len(golds)


def brier_score(prob_maps: Sequence[Dict[str, float]], golds: Sequence[str]) -> float:
    if not golds:
        return float("nan")
    tot = 0.0
    for m, g in zip(prob_maps, golds):
        keys = set(m) | {g}
        tot += sum((m.get(k, 0.0) - (1.0 if k == g else 0.0)) ** 2 for k in keys)
    return tot / len(golds)


def negative_log_likelihood(
    prob_maps: Sequence[Dict[str, float]], golds: Sequence[str], eps: float = 1e-12
) -> float:
    if not golds:
        return float("nan")
    tot = 0.0
    for m, g in zip(prob_maps, golds):
        p = max(eps, float(m.get(g, 0.0)))
        tot += -math.log(p)
    return tot / len(golds)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> float:
    n = len(confidences)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i
            for i, c in enumerate(confidences)
            if (lo < c <= hi) or (b == 0 and c == 0.0)
        ]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(acc - conf)
    return ece


def score_task_metrics(
    predicted_scores: Sequence[float], gold_scores: Sequence[float]
) -> Dict[str, float]:
    """MAE, RMSE, and Pearson correlation for ordinal ``score`` tasks (e.g., SST5)."""
    n = len(gold_scores)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan")}
    diffs = [p - g for p, g in zip(predicted_scores, gold_scores)]
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    mean_p = sum(predicted_scores) / n
    mean_g = sum(gold_scores) / n
    num = sum((p - mean_p) * (g - mean_g) for p, g in zip(predicted_scores, gold_scores))
    den_p = math.sqrt(sum((p - mean_p) ** 2 for p in predicted_scores))
    den_g = math.sqrt(sum((g - mean_g) ** 2 for g in gold_scores))
    pearson = num / (den_p * den_g) if (den_p > 0 and den_g > 0) else 0.0
    return {"mae": mae, "rmse": rmse, "pearson": pearson}


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated percentile for ``q in [0, 100]``."""
    if not values:
        return float("nan")
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def family_bootstrap_ci(
    family_values: Dict[str, float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    agg_fn: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / max(1, len(xs)),
) -> Dict[str, float]:
    """Bootstrap confidence interval grouped by canonical family/query ID (AGENTS.md 9).

    Never treats reordered copies as independent observations.
    """
    fams = sorted(family_values.keys())
    if not fams:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_families": 0}
    vals = [family_values[f] for f in fams]
    point = float(agg_fn(vals))
    if len(vals) == 1 or n_boot <= 0:
        return {"point": point, "ci_low": point, "ci_high": point, "n_families": len(vals)}
    rng = random.Random(seed)
    m = len(vals)
    draws: List[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(m)] for _ in range(m)]
        draws.append(float(agg_fn(sample)))
    return {
        "point": point,
        "ci_low": percentile(draws, 100.0 * (alpha / 2.0)),
        "ci_high": percentile(draws, 100.0 * (1.0 - alpha / 2.0)),
        "n_families": m,
    }

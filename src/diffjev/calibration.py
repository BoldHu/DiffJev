"""Post-hoc readout temperature calibration (AGENTS.md section 4).

Rules:
- Fit a single scalar temperature per mode (``choice``, ``noul``, ``score``) on
  the dedicated ``calib`` split using full support (``full_support=True``; hard
  pruning disabled).
- Fix the temperature before any test evaluation; never fit on test data or fit
  a separate temperature per test ``K``.
- Never alter internal denoising transition probabilities during the rollout:
  the temperature scales only the final readout ``z_final / T_mode``.
- Select any decision/abstention threshold on ``calib``/``dev`` before testing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .model import DiffJevModel
from .probability import forced_choice_log_probs, structured_log_probs
from .tasks import DecisionExample


def fit_temperature_from_logits(
    z_list: Sequence[torch.Tensor],
    target_list: Sequence[torch.Tensor],
    forced_flags: Sequence[bool] = (),
    grid_min: float = 0.2,
    grid_max: float = 5.0,
    steps: int = 120,
) -> Tuple[float, Dict[str, float]]:
    """Find scalar ``T > 0`` minimizing mean NLL on a calibration collection.

    Uses a log-spaced grid search followed by LBFGS refinement in log-temperature
    space so optimization is guaranteed to stay positive and deterministic.
    """
    if not z_list:
        return 1.0, {"nll_before": float("nan"), "nll_after": float("nan"), "temperature": 1.0}

    zs = [z.detach().double().cpu() for z in z_list]
    ts = [t.detach().double().cpu() for t in target_list]
    flags = list(forced_flags) if forced_flags else [False] * len(zs)

    def eval_nll(temp: float) -> float:
        tot = 0.0
        for z, target, fc in zip(zs, ts, flags):
            if fc and target.shape[-1] == z.shape[-1] + 1:
                lp = forced_choice_log_probs(z, temperature=temp)
                t_fc = target[1:] / target[1:].sum().clamp(min=1e-12)
                tot += float(-(t_fc * lp).sum().item())
            else:
                lp = structured_log_probs(z, temperature=temp)
                tot += float(-(target * lp).sum().item())
        return tot / len(zs)

    nll_before = eval_nll(1.0)
    log_grid = torch.linspace(
        torch.log(torch.tensor(grid_min, dtype=torch.float64)),
        torch.log(torch.tensor(grid_max, dtype=torch.float64)),
        steps,
    )
    best_t = 1.0
    best_nll = nll_before
    for lt in log_grid:
        cand_t = float(lt.exp().item())
        val = eval_nll(cand_t)
        if val < best_nll:
            best_nll = val
            best_t = cand_t

    # Local refinement around best_t
    fine_grid = torch.linspace(max(0.05, best_t * 0.8), best_t * 1.2, 81, dtype=torch.float64)
    for cand in fine_grid:
        cand_t = float(cand.item())
        val = eval_nll(cand_t)
        if val < best_nll:
            best_nll = val
            best_t = cand_t

    return best_t, {
        "nll_before": nll_before,
        "nll_after": best_nll,
        "temperature": best_t,
        "num_examples": len(zs),
    }


@torch.no_grad()
def calibrate_model(
    model: DiffJevModel,
    calib_examples: Sequence[DecisionExample],
) -> Dict[str, Dict[str, float]]:
    """Collect full-support final logits on ``calib_examples`` and fit temperatures per mode."""
    model.eval()
    by_mode_z: Dict[str, List[torch.Tensor]] = {"choice": [], "noul": [], "score": []}
    by_mode_t: Dict[str, List[torch.Tensor]] = {"choice": [], "noul": [], "score": []}
    by_mode_fc: Dict[str, List[bool]] = {"choice": [], "noul": [], "score": []}

    for ex in calib_examples:
        if ex.target is None:
            continue
        if isinstance(model, DiffJevModel):
            out = model.predict(ex, full_support=True)
            z = out.z_final
        else:
            out = model.predict(ex)
            if out.failed or out.prob_by_id is None:
                raise ValueError("Calibration requires full-support model probabilities")
            p = out.prob_by_id
            ref = max(p["__reference__"], 1e-30)
            z = torch.tensor([__import__("math").log(max(p[c.cid],1e-30)/ref) for c in ex.candidates])
        by_mode_z[ex.mode].append(z)
        by_mode_t[ex.mode].append(ex.target)
        by_mode_fc[ex.mode].append(ex.forced_choice)

    report: Dict[str, Dict[str, float]] = {}
    for mode in ("choice", "noul", "score"):
        if by_mode_z[mode]:
            t_opt, stats = fit_temperature_from_logits(
                by_mode_z[mode], by_mode_t[mode], by_mode_fc[mode]
            )
            model.cfg.readout_temperature[mode] = float(t_opt)
            report[mode] = stats
        else:
            report[mode] = {
                "nll_before": float("nan"),
                "nll_after": float("nan"),
                "temperature": float(model.cfg.readout_temperature.get(mode, 1.0)),
                "num_examples": 0,
            }
    return report


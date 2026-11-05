"""Academic visualization and publication-grade table formatting (NeurIPS / ICML / ACL style).

Generates vector PDF (``pdf.fonttype = 42``) and 300-DPI PNG figures alongside
GitHub-Markdown, CSV, and LaTeX ``booktabs`` tables, saving all artifacts into
``$DIFFJEV_ROOT/outputs/{figures,tables}`` so they persist on Google Drive or
the A100 server.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import ExperimentLogger, project_paths

# Colorblind-safe Okabe-Ito academic palette
PALETTE: Dict[str, str] = {
    "diffjev": "#0072B2",              # Deep Ocean Blue
    "diffjev_t3_pruning": "#0072B2",
    "full": "#0072B2",
    "diffjev_no_pruning": "#56B4E9",   # Sky Blue
    "no_pruning": "#56B4E9",
    "serialized_dlm": "#D55E00",       # Vermilion
    "serialized_ar": "#009E73",        # Bluish Green
    "llm_as_jev": "#009E73",
    "full_context_t1_control": "#CC79A7",  # Reddish Purple
    "t1_slot_only": "#E69F00",         # Amber / Orange
    "no_cross_slot": "#F0E442",        # Gold
    "untied_positions": "#8C564B",     # Brown
    "no_augmentation": "#7F7F7F",      # Neutral Gray
    "bm25": "#E69F00",
    "bge_biencoder": "#56B4E9",
    "jeva_4b": "#9467BD",
    "jev_api": "#333333",
}

MARKERS: Dict[str, str] = {
    "diffjev": "o",
    "diffjev_t3_pruning": "o",
    "diffjev_no_pruning": "s",
    "serialized_dlm": "^",
    "serialized_ar": "D",
    "llm_as_jev": "D",
    "full_context_t1_control": "v",
    "bm25": "P",
    "bge_biencoder": "X",
}


def _color_for(name: str, idx: int = 0) -> str:
    key = name.lower()
    if key in PALETTE:
        return PALETTE[key]
    for prefix, col in PALETTE.items():
        if prefix in key:
            return col
    fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F7F7F"]
    return fallback[idx % len(fallback)]


def _marker_for(name: str, idx: int = 0) -> str:
    key = name.lower()
    if key in MARKERS:
        return MARKERS[key]
    for prefix, mk in MARKERS.items():
        if prefix in key:
            return mk
    fallback = ["o", "s", "^", "D", "v", "P", "X"]
    return fallback[idx % len(fallback)]


def set_publication_style():
    """Configure Matplotlib rcParams for top-tier academic conference figures."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_and_show(fig, stem: str, out_dir: Optional[Path] = None, show: bool = True) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    target = Path(out_dir) if out_dir is not None else project_paths()["figures"]
    target.mkdir(parents=True, exist_ok=True)
    pdf_path = target / f"{stem}.pdf"
    png_path = target / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    if show:
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            pass
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path), "pdf_path": str(pdf_path), "png_path": str(png_path)}


# ---------------------------------------------------------------------------
# Publication Table Formatting (Markdown + LaTeX booktabs + CSV)
# ---------------------------------------------------------------------------


def _fmt_cell(val: Any, float_fmt: str = ".4f") -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, int):
        return f"{val:,d}"
    if isinstance(val, float):
        if math.isnan(val):
            return "—"
        return format(val, float_fmt)
    return str(val)


def format_markdown_table(
    rows: Sequence[Dict[str, Any]],
    columns: Optional[Sequence[Tuple[str, str]]] = None,
    float_fmt: str = ".4f",
) -> str:
    """Format ``rows`` as a GitHub-flavored Markdown table.

    ``columns``: optional list of ``(dict_key, display_header)``.
    """
    if not rows:
        return "_No records._"
    if columns is None:
        columns = [(k, k) for k in rows[0].keys()]
    headers = [h for _, h in columns]
    keys = [k for k, _ in columns]
    str_rows = [[_fmt_cell(r.get(k), float_fmt) for k in keys] for r in rows]
    widths = [
        max(len(headers[c]), max((len(sr[c]) for sr in str_rows), default=1))
        for c in range(len(columns))
    ]
    header_line = "| " + " | ".join(headers[c].ljust(widths[c]) for c in range(len(columns))) + " |"
    sep_line = "| " + " | ".join("-" * widths[c] for c in range(len(columns))) + " |"
    body_lines = [
        "| " + " | ".join(sr[c].ljust(widths[c]) for c in range(len(columns))) + " |"
        for sr in str_rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def format_latex_booktabs_table(
    rows: Sequence[Dict[str, Any]],
    columns: Optional[Sequence[Tuple[str, str]]] = None,
    caption: str = "Experimental Results",
    label: str = "tab:results",
    float_fmt: str = ".4f",
) -> str:
    """Format ``rows`` as a publication-ready LaTeX ``booktabs`` table."""
    if not rows:
        return "% Empty table"
    if columns is None:
        columns = [(k, k) for k in rows[0].keys()]
    headers = [h.replace("_", r"\_").replace("%", r"\%") for _, h in columns]
    keys = [k for k, _ in columns]
    align = "l" + "r" * max(0, len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [
            _fmt_cell(r.get(k), float_fmt).replace("_", r"\_").replace("%", r"\%")
            for k in keys
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def render_and_save_table(
    logger: ExperimentLogger,
    table_name: str,
    rows: Sequence[Dict[str, Any]],
    columns: Optional[Sequence[Tuple[str, str]]] = None,
    caption: str = "Experimental Results",
    float_fmt: str = ".4f",
) -> Dict[str, str]:
    """Print Markdown table, save CSV + LaTeX booktabs table to ``outputs/tables``."""
    md_str = format_markdown_table(rows, columns=columns, float_fmt=float_fmt)
    tex_str = format_latex_booktabs_table(
        rows,
        columns=columns,
        caption=caption,
        label=f"tab:{table_name}",
        float_fmt=float_fmt,
    )
    csv_path = logger.save_csv(table_name, rows, columns=[k for k, _ in columns] if columns else None)
    tex_path = logger.save_latex(table_name, tex_str)
    print(f"\n### Table: {caption}\n{md_str}\n")
    return {"markdown": md_str, "csv": str(csv_path), "latex": str(tex_path)}


# ---------------------------------------------------------------------------
# Stage & Experiment Figure Generators
# ---------------------------------------------------------------------------


def plot_data_preparation_summary(
    manifest_dict: Dict[str, Any],
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 01 figure: (a) 4-bucket split counts, (b) Realized K histogram."""
    plt = set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # Panel (a): Grouped bar chart of splits per bucket
    b_counts = manifest_dict.get("bucket_counts", {})
    buckets = ["openjev_choice", "openjev_noul", "openjev_score", "toolret_choice"]
    short_labels = ["OpenJev\nChoice", "OpenJev\nNoul", "OpenJev\nScore", "ToolRet\nChoice"]
    splits = ["train", "dev", "calib", "quarantine"]
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
    width = 0.18
    xs = list(range(len(buckets)))
    for s_i, sp in enumerate(splits):
        vals = [int(b_counts.get(sp, {}).get(b, 0)) for b in buckets]
        offset = (s_i - 1.5) * width
        ax1.bar([x + offset for x in xs], vals, width=width, label=sp.capitalize(), color=colors[s_i], edgecolor="black", linewidth=0.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(short_labels)
    ax1.set_ylabel("Number of Source Records")
    ax1.set_title("(a) Stratified Bucket Splits & Overlap Quarantine")
    ax1.legend(loc="upper right")

    # Panel (b): Realized candidate count K histogram across the training trace
    k_hist = manifest_dict.get("candidate_k_histogram", {})
    ks = sorted(int(k) for k in k_hist.keys())
    counts = [int(k_hist.get(k, k_hist.get(str(k), 0))) for k in ks]
    x_pos = list(range(len(ks)))
    ax2.bar(x_pos, counts, color="#0072B2", alpha=0.85, edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel(r"Realized Candidate Count $K$ ($K \leq 32$ Training Cap)")
    ax2.set_ylabel("Example Presentations")
    ax2.set_title(r"(b) Realized $K$ Distribution in Training Trace")

    fig.tight_layout()
    return _save_and_show(fig, "fig01_data_preparation", out_dir=out_dir, show=show)


def plot_training_and_calibration(
    history: Sequence[Dict[str, Any]],
    calib_report: Dict[str, Dict[str, Any]],
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 02 figure: (a) Composite training losses, (b) Pre/Post calibration NLL."""
    plt = set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    steps = [int(r.get("step", i)) + 1 for i, r in enumerate(history)]
    for key, label, col, ls in [
        ("total", r"$\mathcal{L}_{\mathrm{total}}$", "#0072B2", "-"),
        ("L_mask", r"$\mathcal{L}_{\mathrm{mask}}$", "#D55E00", "--"),
        ("L_task_CE", r"$\mathcal{L}_{\mathrm{task\_CE}}$", "#009E73", "-."),
        ("L_Brier", r"$\mathcal{L}_{\mathrm{Brier}}$", "#CC79A7", ":"),
    ]:
        if any(key in r for r in history):
            ys = [float(r.get(key, float("nan"))) for r in history]
            ax1.plot(steps, ys, label=label, color=col, linestyle=ls, marker="o" if len(steps) <= 30 else None, markersize=4)
    ax1.set_xlabel("Optimizer Update Step")
    ax1.set_ylabel("Loss Value")
    ax1.set_title("(a) Structured Co-Denoising Training Trajectory")
    ax1.legend(loc="best")

    modes = [m for m in ("choice", "noul", "score") if m in calib_report]
    before = [float(calib_report[m].get("nll_before", 0.0) or 0.0) for m in modes]
    after = [float(calib_report[m].get("nll_after", 0.0) or 0.0) for m in modes]
    temps = [float(calib_report[m].get("temperature", 1.0) or 1.0) for m in modes]
    xs = list(range(len(modes)))
    w = 0.32
    ax2.bar([x - w / 2 for x in xs], before, width=w, label=r"Uncalibrated ($T=1.0$)", color="#7F7F7F", edgecolor="black", linewidth=0.6)
    bars_after = ax2.bar([x + w / 2 for x in xs], after, width=w, label=r"Calibrated ($T=T_{\mathrm{mode}}^*$)", color="#0072B2", edgecolor="black", linewidth=0.6)
    for rect, t_val in zip(bars_after, temps):
        h = rect.get_height()
        if not math.isnan(h):
            ax2.annotate(
                f"$T^*={t_val:.2f}$",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([m.capitalize() for m in modes])
    ax2.set_ylabel(r"Calibration Split NLL ($\downarrow$)")
    ax2.set_title("(b) Full-Support Post-Hoc Temperature Calibration")
    ax2.legend(loc="best")

    fig.tight_layout()
    return _save_and_show(fig, "fig02_training_calibration", out_dir=out_dir, show=show)


def plot_exp1_benchmark(
    reports: Sequence[Any],
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 04 (Exp 1) figure: (a) Aggregate Accuracy + 95% CI, (b) Calibration Brier & ECE."""
    plt = set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))

    names = [r.model_name for r in reports]
    accs = [float(r.aggregate.get("accuracy") or 0.0) * 100.0 for r in reports]
    cc_accs = [float(r.aggregate.get("chance_corrected_accuracy") or 0.0) * 100.0 for r in reports]
    ci_lows = [
        max(0.0, accs[i] - float((r.aggregate.get("family_bootstrap_accuracy") or {}).get("ci_low", accs[i] / 100.0)) * 100.0)
        for i, r in enumerate(reports)
    ]
    ci_highs = [
        max(0.0, float((r.aggregate.get("family_bootstrap_accuracy") or {}).get("ci_high", accs[i] / 100.0)) * 100.0 - accs[i])
        for i, r in enumerate(reports)
    ]

    xs = list(range(len(names)))
    w = 0.34
    ax1.bar(
        [x - w / 2 for x in xs],
        accs,
        yerr=[ci_lows, ci_highs],
        capsize=3,
        width=w,
        label="Accuracy (%)",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.bar(
        [x + w / 2 for x in xs],
        cc_accs,
        width=w,
        label="Chance-Corrected Acc. (%)",
        color="#009E73",
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel(r"Score (%) $\uparrow$")
    ax1.set_title("(a) Exp 1: JevBench Aggregate Accuracy (95% Family CI)")
    ax1.legend(loc="best")

    briers = [float(r.aggregate.get("brier") or 0.0) for r in reports]
    eces = [float(r.aggregate.get("ece") or 0.0) for r in reports]
    ax2.bar([x - w / 2 for x in xs], briers, width=w, label=r"Brier Score ($\downarrow$)", color="#D55E00", edgecolor="black", linewidth=0.6)
    ax2.bar([x + w / 2 for x in xs], eces, width=w, label=r"ECE ($\downarrow$)", color="#CC79A7", edgecolor="black", linewidth=0.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel(r"Calibration Error ($\downarrow$)")
    ax2.set_title("(b) Exp 1: Probabilistic Calibration (Brier & ECE)")
    ax2.legend(loc="best")

    fig.tight_layout()
    return _save_and_show(fig, "fig04_exp1_benchmark", out_dir=out_dir, show=show)


def plot_exp2_robustness(
    reports: Sequence[Any],
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 04 (Exp 2) figure: (a) Order & Alias Decision Flip Rate, (b) Probability Drift."""
    plt = set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))

    names = [r.model_name for r in reports]
    order_dfr = [float(r.order_permutation.get("decision_flip_rate") or 0.0) * 100.0 for r in reports]
    alias_dfr = [float(r.alias_renaming.get("decision_flip_rate") or 0.0) * 100.0 for r in reports]
    order_drift = [float(r.order_permutation.get("mean_max_prob_drift") or 0.0) for r in reports]
    alias_drift = [float(r.alias_renaming.get("mean_max_prob_drift") or 0.0) for r in reports]

    xs = list(range(len(names)))
    w = 0.34
    ax1.bar([x - w / 2 for x in xs], order_dfr, width=w, label="Option-Order DFR (%)", color="#0072B2", edgecolor="black", linewidth=0.6)
    ax1.bar([x + w / 2 for x in xs], alias_dfr, width=w, label="Alias-Renaming DFR (%)", color="#E69F00", edgecolor="black", linewidth=0.6)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel(r"Decision Flip Rate (%) $\downarrow$")
    ax1.set_title("(a) Exp 2: Semantic Decision Flip Rate (Order vs. Alias)")
    ax1.legend(loc="best")

    ax2.bar([x - w / 2 for x in xs], order_drift, width=w, label="Order Max Prob. Drift", color="#D55E00", edgecolor="black", linewidth=0.6)
    ax2.bar([x + w / 2 for x in xs], alias_drift, width=w, label="Alias Max Prob. Drift", color="#CC79A7", edgecolor="black", linewidth=0.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel(r"Mean $\max_c |\Delta P(c)|$ ($\downarrow$)")
    ax2.set_title("(b) Exp 2: Inverse-Permuted Probability Drift")
    ax2.legend(loc="best")

    fig.tight_layout()
    return _save_and_show(fig, "fig04_exp2_robustness", out_dir=out_dir, show=show)


def plot_exp3_scaling(
    hard_reports: Sequence[Any],
    random_reports: Sequence[Any] = (),
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 05 (Exp 3) figure: (a) Hit@1 vs K (BM25-Hard), (b) Stagewise Gold Survival."""
    plt = set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1))

    for idx, rep in enumerate(hard_reports):
        ks = sorted(int(k) for k in rep.by_k.keys())
        ys = [float(rep.by_k[k]["hit_at_1"]) * 100.0 for k in ks]
        col = _color_for(rep.model_name, idx)
        mk = _marker_for(rep.model_name, idx)
        ax1.plot(ks, ys, label=f"{rep.model_name} (BM25-Hard)", color=col, marker=mk)

    for idx, rep in enumerate(random_reports):
        ks = sorted(int(k) for k in rep.by_k.keys())
        ys = [float(rep.by_k[k]["hit_at_1"]) * 100.0 for k in ks]
        col = _color_for(rep.model_name, idx)
        ax1.plot(ks, ys, label=f"{rep.model_name} (Random)", color=col, linestyle="--", marker="o", alpha=0.75)

    ax1.axvline(32, color="#444444", linestyle=":", linewidth=1.1, label=r"Train Cap ($K=32$)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Candidate Count $K$ ($\log_2$ scale)")
    ax1.set_ylabel(r"Routing Accuracy Hit@1 (%) $\uparrow$")
    ax1.set_title("(a) Exp 3: Long-Tail Candidate-Count Scaling")
    ax1.legend(loc="best")

    # Panel (b): Stagewise Gold Survival for DiffJev across passes
    diffjev_rep = next((r for r in hard_reports if r.by_k and any(r.by_k[k].get("stagewise_gold_survival") for k in r.by_k)), None)
    if diffjev_rep is not None:
        ks = sorted(int(k) for k in diffjev_rep.by_k.keys())
        max_stages = max((len(diffjev_rep.by_k[k].get("stagewise_gold_survival") or []) for k in ks), default=0)
        stage_names = [r"Pass 1 Input ($K$)", r"Pass 2 Survivors ($\leq 32$)", r"Pass 3 Survivors ($\leq 8$)", "Final Retained"]
        stage_cols = ["#7F7F7F", "#56B4E9", "#009E73", "#0072B2"]
        for s_idx in range(max_stages):
            ys = [
                float(diffjev_rep.by_k[k]["stagewise_gold_survival"][s_idx]) * 100.0
                for k in ks
                if len(diffjev_rep.by_k[k].get("stagewise_gold_survival") or []) > s_idx
            ]
            if ys:
                ax2.plot(
                    ks[: len(ys)],
                    ys,
                    label=stage_names[s_idx] if s_idx < len(stage_names) else f"Stage {s_idx+1}",
                    color=stage_cols[s_idx % len(stage_cols)],
                    marker="s",
                )
    ax2.axvline(32, color="#444444", linestyle=":", linewidth=1.1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Candidate Count $K$ ($\log_2$ scale)")
    ax2.set_ylabel(r"Gold Candidate Survival (%) $\uparrow$")
    ax2.set_title("(b) Exp 3: Progressive Pruning Stagewise Gold Survival")
    ax2.legend(loc="best")

    fig.tight_layout()
    return _save_and_show(fig, "fig05_exp3_scaling", out_dir=out_dir, show=show)


def plot_exp4_ablations(
    ablation_table: Sequence[Dict[str, Any]],
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 06 (Exp 4) figure: (a) Single-factor ablation metrics, (b) Accuracy vs GFLOPs."""
    plt = set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))

    variants = [r["variant"] for r in ablation_table]
    bench_acc = [float(r.get("benchmark_accuracy") or 0.0) * 100.0 for r in ablation_table]
    dfr = [float(r.get("order_flip_rate") or 0.0) * 100.0 for r in ablation_table]
    gflops = [float(r.get("infer_gflops_at_kmax") or 0.0) for r in ablation_table]

    ys = list(range(len(variants)))
    h = 0.35
    ax1.barh([y + h / 2 for y in ys], bench_acc, height=h, label=r"Benchmark Acc. (%) $\uparrow$", color="#0072B2", edgecolor="black", linewidth=0.6)
    ax1.barh([y - h / 2 for y in ys], dfr, height=h, label=r"Order Flip Rate (%) $\downarrow$", color="#D55E00", edgecolor="black", linewidth=0.6)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(variants)
    ax1.invert_yaxis()
    ax1.set_xlabel("Percentage (%)")
    ax1.set_title("(a) Exp 4: Controlled Single-Factor Ablations")
    ax1.legend(loc="best")

    for i, r in enumerate(ablation_table):
        v = r["variant"]
        col = _color_for(v, i)
        mk = _marker_for(v, i)
        ax2.scatter(gflops[i], bench_acc[i], color=col, marker=mk, s=75, edgecolors="black", linewidths=0.7, label=v)
        ax2.annotate(v, (gflops[i], bench_acc[i]), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax2.set_xlabel(r"Full-Request Inference Compute at $K_{\max}$ (GFLOPs) $\downarrow$")
    ax2.set_ylabel(r"Benchmark Accuracy (%) $\uparrow$")
    ax2.set_title("(b) Exp 4: Accuracy vs. Inference Compute Pareto Frontier")

    fig.tight_layout()
    return _save_and_show(fig, "fig06_exp4_ablations", out_dir=out_dir, show=show)


def plot_exp5_efficiency(
    panel_result: Dict[str, Any],
    out_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, str]:
    """Notebook 07 (Exp 5) 4-panel figure required by AGENTS.md line 241:
    (a) Latency p50/p95 vs K, (b) Operator-shape GFLOPs vs K,
    (c) Peak GPU Memory vs K, (d) Matched-Query Speedup vs K."""
    plt = set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    ax_lat, ax_flops, ax_mem, ax_spd = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    rows = panel_result.get("table", [])
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    for idx, m in enumerate(methods):
        m_rows = sorted([r for r in rows if r["method"] == m and r.get("coverage", 0) > 0], key=lambda x: x["K"])
        if not m_rows:
            continue
        ks = [int(r["K"]) for r in m_rows]
        p50 = [float(r["wall_p50_ms"]) for r in m_rows]
        p95 = [float(r["wall_p95_ms"]) for r in m_rows]
        gf = [float(r["mean_gflops"]) for r in m_rows]
        mem = [float(r["peak_memory_mb"]) for r in m_rows]
        col = _color_for(m, idx)
        mk = _marker_for(m, idx)

        ax_lat.plot(ks, p50, label=f"{m} ($p_{{50}}$)", color=col, marker=mk)
        ax_lat.fill_between(ks, p50, p95, color=col, alpha=0.14)

        ax_flops.plot(ks, gf, label=m, color=col, marker=mk)
        ax_mem.plot(ks, mem, label=m, color=col, marker=mk)

        spd_dlm = [r.get("speedup_vs_serialized_dlm") for r in m_rows]
        valid_spd = [(k, float(s)) for k, s in zip(ks, spd_dlm) if s is not None]
        if valid_spd:
            ax_spd.plot([x[0] for x in valid_spd], [x[1] for x in valid_spd], label=f"{m} vs. Ser-DLM", color=col, marker=mk)

    ax_lat.set_xscale("log", base=2)
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel(r"Candidate Count $K$ ($\log_2$ scale)")
    ax_lat.set_ylabel(r"Request Wall Latency (ms, $p_{50}$ & $p_{95}$ band) $\downarrow$")
    ax_lat.set_title(r"(a) Exp 5: End-to-End Request Latency vs. $K$")
    ax_lat.legend(loc="best")

    ax_flops.set_xscale("log", base=2)
    ax_flops.set_yscale("log")
    ax_flops.set_xlabel(r"Candidate Count $K$ ($\log_2$ scale)")
    ax_flops.set_ylabel(r"Full-Request Compute (GFLOPs, 2 FLOPs/MAC) $\downarrow$")
    ax_flops.set_title(r"(b) Exp 5: Operator-Shape GFLOPs vs. $K$")
    ax_flops.legend(loc="best")

    ax_mem.set_xscale("log", base=2)
    ax_mem.set_xlabel(r"Candidate Count $K$ ($\log_2$ scale)")
    ax_mem.set_ylabel(r"Peak Allocated GPU Memory (MB) $\downarrow$")
    ax_mem.set_title(r"(c) Exp 5: Peak Resident & Activation Memory vs. $K$")
    ax_mem.legend(loc="best")

    ax_spd.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    ax_spd.set_xscale("log", base=2)
    ax_spd.set_xlabel(r"Candidate Count $K$ ($\log_2$ scale)")
    ax_spd.set_ylabel(r"Matched-Query Speedup ($\times$ vs. Serialized DLM) $\uparrow$")
    ax_spd.set_title("(d) Exp 5: Speedup on Shared Successful Queries")
    ax_spd.legend(loc="best")

    fig.tight_layout()
    size_tag = panel_result.get("size", "panel")
    return _save_and_show(fig, f"fig07_exp5_efficiency_{size_tag}", out_dir=out_dir, show=show)

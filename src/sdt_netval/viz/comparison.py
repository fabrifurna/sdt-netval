"""Publication-ready comparison of a baseline against experimental conditions.

One panel per metric; bars show the mean over the runs of each condition, error
bars the 95% confidence interval (t-distribution), dots the individual runs and
a dashed line the baseline mean. Figures are exported as PNG (raster) and PDF
(vector) so they can be dropped straight into a LaTeX document.

matplotlib and seaborn are imported here, not in the package root, so that
``import sdt_netval`` stays light for users who only need the metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from sdt_netval.core.stats import mean_ci95

logger = logging.getLogger(__name__)

BASELINE_KEY = "__baseline__"

DEFAULT_METRICS: Dict[str, str] = {
    "alpha_in_degree": "Power-law exponent (α)",
    "modularity": "Modularity (Q)",
    "average_clustering": "Average clustering",
}

# Categorical palette in fixed slot order (blue, orange, aqua, yellow, magenta,
# green, violet, red); the baseline is always drawn in neutral grey.
_PALETTE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
_BASELINE_COLOR = "#898781"

_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"


def plot_baseline_comparison(
    baseline: pd.DataFrame,
    data: pd.DataFrame,
    output: Optional[Union[str, Path]] = None,
    *,
    metrics: Optional[Mapping[str, str]] = None,
    condition_col: str = "condition",
    conditions: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    baseline_label: str = "Baseline",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    figsize: Optional[tuple] = None,
    caption: Optional[str] = None,
) -> Figure:
    """Draw a baseline-vs-conditions figure, one panel per metric.

    Args:
        baseline: Per-run table of the baseline (e.g. the raw Stage A table).
        data: Per-run table of the experimental conditions (e.g. the raw Stage B
            table), with a ``condition_col`` column.
        output: Optional path *without extension*; one file per entry of ``formats``
            is written next to it (``<output>.png``, ``<output>.pdf``).
        metrics: Mapping ``column -> panel title``. Defaults to α, Q and clustering.
        condition_col: Condition column of ``data``.
        conditions: Conditions to show, in order. Defaults to appearance order.
        labels: Optional ``condition -> display label`` (use ``\\n`` to wrap).
        colors: Optional ``condition -> hex color``. Unspecified conditions take the
            categorical palette in fixed order; the baseline is always grey.
        baseline_label: Display label of the baseline bar.
        formats: File formats to export when ``output`` is given.
        dpi: Resolution of raster exports.
        figsize: Figure size in inches; defaults to a width proportional to the panels.
        caption: Text under the figure; a sensible default explains the encodings.

    Returns:
        The matplotlib Figure (left open; call ``plt.close(fig)`` when done).

    Raises:
        ValueError: If a metric column is missing, ``data`` has no conditions, or more
            than eight conditions are requested.
    """
    metrics = dict(metrics) if metrics is not None else dict(DEFAULT_METRICS)
    labels = dict(labels or {})
    colors = dict(colors or {})

    if condition_col not in data.columns:
        raise ValueError(f"Column '{condition_col}' not found in data.")
    order = list(conditions) if conditions is not None else list(dict.fromkeys(data[condition_col]))
    if not order:
        raise ValueError("No conditions to plot.")
    if len(order) > len(_PALETTE):
        raise ValueError(
            f"At most {len(_PALETTE)} conditions can be drawn with distinct colors; "
            "pass a subset via `conditions`."
        )
    for name in metrics:
        for label, df in (("baseline", baseline), ("data", data)):
            if name not in df.columns:
                raise ValueError(f"Metric '{name}' not found in the {label} table.")

    samples = {BASELINE_KEY: baseline}
    samples.update({c: data[data[condition_col] == c] for c in order})
    empty = [c for c, df in samples.items() if df.empty]
    if empty:
        names = ["baseline" if c == BASELINE_KEY else c for c in empty]
        raise ValueError(f"No runs found for: {names}")

    keys = [BASELINE_KEY] + order
    display = {BASELINE_KEY: baseline_label, **{c: labels.get(c, c) for c in order}}
    palette = {c: colors.get(c, _PALETTE[i]) for i, c in enumerate(order)}
    palette[BASELINE_KEY] = _BASELINE_COLOR

    with sns.axes_style("whitegrid", rc={
        "axes.facecolor": _SURFACE, "figure.facecolor": _SURFACE, "grid.color": _GRID,
        "axes.edgecolor": _GRID, "axes.labelcolor": _INK_SECONDARY,
        "xtick.color": _INK_SECONDARY, "ytick.color": _INK_SECONDARY, "text.color": _INK,
    }), plt.rc_context({"pdf.fonttype": 42}):  # embed TrueType: selectable text in LaTeX
        fig, axes = plt.subplots(
            1, len(metrics), figsize=figsize or (4.3 * len(metrics) + 0.4, 4.6), squeeze=False
        )
        rng = np.random.default_rng(42)

        for ax, (metric, title) in zip(axes[0], metrics.items(), strict=True):
            xs = np.arange(len(keys))
            summ = [mean_ci95(samples[k][metric]) for k in keys]
            means = np.array([s["mean"] for s in summ])
            lo = np.array([s["ci_low"] for s in summ])
            hi = np.array([s["ci_high"] for s in summ])

            ax.bar(xs, means, width=0.62, color=[palette[k] for k in keys],
                   edgecolor=_SURFACE, linewidth=1.5, zorder=2)
            ax.errorbar(xs, means, yerr=[means - lo, hi - means], fmt="none", ecolor=_INK,
                        elinewidth=1.4, capsize=4, capthick=1.4, zorder=4)

            run_max = []
            for x, k in zip(xs, keys, strict=True):
                y = samples[k][metric].dropna().to_numpy(dtype=float)
                run_max.append(y.max())
                ax.scatter(x + rng.uniform(-0.16, 0.16, len(y)), y, s=12, color=_INK,
                           alpha=0.35, linewidths=0, zorder=3)

            # Mean labels sit above the highest run so they never collide with the dots
            tops = np.maximum(np.array(run_max), hi)
            for x, m, top in zip(xs, means, tops, strict=True):
                ax.text(x, top + 0.025 * tops.max(), f"{m:.2f}", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color=_INK, zorder=5)

            ax.axhline(means[0], color=_INK_SECONDARY, linestyle="--", linewidth=1.0,
                       alpha=0.8, zorder=1)
            if metric == "alpha_in_degree":
                ax.axhspan(2, 3, color=_PALETTE[0], alpha=0.06, zorder=0)  # scale-free range

            ax.set_title(title, fontsize=12, fontweight="bold", color=_INK, pad=10)
            ax.set_xticks(xs)
            ax.set_xticklabels([display[k] for k in keys], fontsize=9)
            ax.set_ylim(0, None)
            ax.margins(y=0.12)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax, left=True)

        if caption is None:
            caption = "Mean over runs with 95% CI · dots: single runs · dashed line: baseline mean"
            if "alpha_in_degree" in metrics:
                caption += " · shaded band: scale-free range 2 < α < 3"
        fig.suptitle(caption, fontsize=10.5, color=_INK_SECONDARY, y=0.02)
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        if output is not None:
            stem = Path(output).resolve()
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in formats:
                path = stem.parent / f"{stem.name}.{ext.lstrip('.')}"
                kwargs = {"dpi": dpi} if ext.lstrip(".").lower() in ("png", "jpg", "jpeg") else {}
                fig.savefig(path, bbox_inches="tight", facecolor=_SURFACE, **kwargs)
                logger.info("Figure saved: file '%s'", path)

    return fig

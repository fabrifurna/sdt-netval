"""Baseline-vs-condition hypothesis testing for multi-run simulation studies.

Implements the non-parametric comparison used in Stage B: for every metric, the
distribution over the runs of each experimental condition is compared with the
baseline distribution through a Mann-Whitney U test (no normality assumption,
suitable for N ≈ 10–30 runs per group). Because several conditions and metrics are
tested at once, p-values are corrected for multiple comparisons, and a
distribution-free effect size (Cliff's delta) is reported next to each p-value.
"""

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from sdt_netval.core.metrics import KEY_METRICS

logger = logging.getLogger(__name__)

_CORRECTIONS = ("holm", "bonferroni", "none")

# Romano et al. (2006) magnitude thresholds for |Cliff's delta|
_DELTA_THRESHOLDS = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))

_COLUMNS = [
    "metric", "condition", "n_baseline", "n_condition", "mean_baseline",
    "mean_condition", "mean_diff", "pct_change", "U", "p_value", "p_adjusted",
    "significance", "cliffs_delta", "effect_magnitude",
]


def significance_flag(p_value: float) -> str:
    """Convert a p-value to the usual academic marker: '***', '**', '*' or 'ns'."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def cliffs_delta(condition: Sequence[float], baseline: Sequence[float]) -> float:
    """Cliff's delta of a condition against the baseline, in [-1, 1].

    Equals P(condition > baseline) − P(condition < baseline): positive when the
    condition tends to produce larger values than the baseline.
    """
    diff = np.asarray(condition, dtype=float)[:, None] - np.asarray(baseline, dtype=float)[None, :]
    return float(np.sign(diff).mean())


def _delta_magnitude(delta: float) -> str:
    a = abs(delta)
    for threshold, name in _DELTA_THRESHOLDS:
        if a < threshold:
            return name
    return "large"


def adjust_pvalues(p_values: Sequence[float], method: str = "holm") -> np.ndarray:
    """Correct a family of p-values for multiple comparisons.

    Args:
        p_values: Raw p-values of one family of tests.
        method: ``'holm'`` (Holm-Bonferroni step-down, uniformly more powerful than
            Bonferroni), ``'bonferroni'``, or ``'none'``.

    Returns:
        Adjusted p-values, in the input order, capped at 1.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in _CORRECTIONS:
        raise ValueError(f"Unknown correction '{method}'. Choose from {_CORRECTIONS}.")
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if method == "none" or m == 0:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)

    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def compare_to_baseline(
    data: pd.DataFrame,
    baseline: Union[str, pd.DataFrame],
    *,
    metrics: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    condition_col: str = "condition",
    baseline_label: str = "baseline",
    correction: str = "holm",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Mann-Whitney U test of every condition against a baseline, for every metric.

    Args:
        data: Per-run table (e.g. ``StageBAnalyzer.full_sensitivity_report()['raw']``)
            with one row per run, a condition column and one column per metric.
        baseline: Either the name of a condition contained in ``data`` (single-table
            layout, e.g. ``'c0'``), or a separate per-run DataFrame holding the
            baseline runs (e.g. the raw Stage A table).
        metrics: Metrics to test. Defaults to the key topological metrics.
        conditions: Conditions to test, in output order. Defaults to all conditions
            of ``data`` other than the baseline.
        condition_col: Name of the condition column in ``data``.
        baseline_label: Label used for an external baseline DataFrame in the logs.
        correction: Multiple-comparison correction applied within each metric across
            conditions: ``'holm'`` (default), ``'bonferroni'`` or ``'none'``.
        alternative: ``'two-sided'``, ``'less'`` or ``'greater'``, expressed for the
            condition relative to the baseline.

    Returns:
        Long-format DataFrame, one row per (metric, condition), with columns
        ``metric, condition, n_baseline, n_condition, mean_baseline, mean_condition,
        mean_diff, pct_change, U, p_value, p_adjusted, significance, cliffs_delta,
        effect_magnitude``. ``U`` is the statistic of the baseline sample;
        ``significance`` is computed on ``p_adjusted``; ``cliffs_delta`` is positive
        when the condition tends to exceed the baseline.

    Raises:
        ValueError: On a missing condition column, an empty baseline, or an unknown option.
    """
    metrics = list(metrics) if metrics is not None else list(KEY_METRICS)
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"Unknown alternative '{alternative}'.")
    if correction not in _CORRECTIONS:
        raise ValueError(f"Unknown correction '{correction}'. Choose from {_CORRECTIONS}.")
    if condition_col not in data.columns:
        raise ValueError(f"Column '{condition_col}' not found in data.")

    if isinstance(baseline, str):
        base_df = data[data[condition_col] == baseline]
        base_name = baseline
        pool = data[data[condition_col] != baseline]
    else:
        base_df, base_name, pool = baseline, baseline_label, data
    if base_df.empty:
        raise ValueError(f"No baseline runs found for '{base_name}'.")

    if conditions is None:
        conditions = list(dict.fromkeys(pool[condition_col]))
    conditions = list(conditions)

    rows: List[Dict[str, object]] = []
    for metric in metrics:
        if metric not in base_df.columns or metric not in pool.columns:
            logger.warning("Metric '%s' not available in both baseline and data. Skipped.", metric)
            continue
        x_base = base_df[metric].dropna().to_numpy(dtype=float)
        if len(x_base) < 2:
            logger.warning("Baseline has <2 valid runs for '%s'. Skipped.", metric)
            continue

        family: List[Dict[str, object]] = []
        for cond in conditions:
            x_cond = pool.loc[pool[condition_col] == cond, metric].dropna().to_numpy(dtype=float)
            if len(x_cond) < 2:
                logger.warning("Condition '%s' has <2 valid runs for '%s'. Skipped.", cond, metric)
                continue

            u_stat, p_val = stats.mannwhitneyu(x_base, x_cond, alternative=_flip(alternative))
            mean_base, mean_cond = float(x_base.mean()), float(x_cond.mean())
            delta = cliffs_delta(x_cond, x_base)
            family.append({
                "metric": metric,
                "condition": cond,
                "n_baseline": len(x_base),
                "n_condition": len(x_cond),
                "mean_baseline": mean_base,
                "mean_condition": mean_cond,
                "mean_diff": mean_cond - mean_base,
                "pct_change": (
                    100.0 * (mean_cond - mean_base) / mean_base if mean_base else np.nan
                ),
                "U": float(u_stat),
                "p_value": float(p_val),
                "cliffs_delta": delta,
                "effect_magnitude": _delta_magnitude(delta),
            })

        adjusted = adjust_pvalues([r["p_value"] for r in family], correction)
        for row, p_adj in zip(family, adjusted, strict=True):
            row["p_adjusted"] = float(p_adj)
            row["significance"] = significance_flag(float(p_adj))
        rows.extend(family)

    logger.info(
        "Compared %d conditions on %d metrics against '%s' (%s correction).",
        len(conditions), len(metrics), base_name, correction,
    )
    return pd.DataFrame(rows, columns=_COLUMNS)


def _flip(alternative: str) -> str:
    # scipy tests the first sample (baseline) relative to the second, whereas the
    # public `alternative` is expressed for the condition relative to the baseline.
    return {"less": "greater", "greater": "less"}.get(alternative, alternative)

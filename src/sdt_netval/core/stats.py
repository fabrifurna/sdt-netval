"""Small statistical helpers shared by the pipelines, analysis and plotting layers."""

from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci95(values: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Mean, sample std and 95% confidence interval (t-distribution) of a sample.

    The t-distribution is the appropriate choice for the small samples (N ≈ 10–30
    runs) typical of simulation studies. NaNs are dropped.

    Args:
        values: Sample of observations.

    Returns:
        Dictionary with keys ``n``, ``mean``, ``std``, ``sem``, ``ci_low``, ``ci_high``.
        With a single observation the interval collapses on the mean.

    Raises:
        ValueError: If the sample contains no finite observation.
    """
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("Cannot summarise an empty sample.")
    mean = float(x.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0, "sem": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(x.std(ddof=1))
    sem = std / float(np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - t_crit * sem,
        "ci_high": mean + t_crit * sem,
    }

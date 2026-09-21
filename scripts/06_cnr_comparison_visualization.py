"""Baseline vs RecSys comparison figure (CNR/Rossetti dataset) — entry point.

Thin wrapper around ``sdt_netval.viz.plot_baseline_comparison``: it only holds the
dataset-specific paths, display labels and colors.

Usage:
    python scripts/06_cnr_comparison_visualization.py
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sdt_netval.core.metrics import KEY_METRICS
from sdt_netval.core.stats import mean_ci95
from sdt_netval.viz import plot_baseline_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting is very chatty
logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "01_processed"
INPUT_STAGE_A = PROCESSED / "02_cnr_baseline_stageA" / "stage_a_raw.csv"
INPUT_STAGE_B = PROCESSED / "03_cnr_recsys_stageB" / "stage_b_raw.csv"
OUTPUT_STEM = PROCESSED / "03_cnr_recsys_stageB" / "cnr_baseline_vs_recsys"
OUTPUT_SUMMARY = PROCESSED / "03_cnr_recsys_stageB" / "cnr_baseline_vs_recsys_summary.csv"

# Display order follows the narrative of the results chapter: organic control
# first, then the two "pathological" recommenders, then the hybrid.
CONDITIONS = ["c3_follower", "c2_collaborative_uu", "c1_popularity", "c4_follower_popularity"]
LABELS = {
    "c3_follower": "Follower",
    "c2_collaborative_uu": "Collaborative\nFiltering",
    "c1_popularity": "Popularity",
    "c4_follower_popularity": "Follower +\nPopularity",
}


def summary_table(baseline: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and 95% CI of every key metric, for the baseline and each condition."""
    rows = []
    groups = [("baseline", baseline)] + [(c, data[data["condition"] == c]) for c in CONDITIONS]
    for name, df in groups:
        for metric in KEY_METRICS:
            rows.append({"condition": name, "metric": metric, **mean_ci95(df[metric])})
    return pd.DataFrame(rows).round(6)


if __name__ == "__main__":
    logger.info("=== CNR — Baseline vs RecSys comparison ===")
    baseline = pd.read_csv(INPUT_STAGE_A)
    data = pd.read_csv(INPUT_STAGE_B)

    fig = plot_baseline_comparison(
        baseline, data, OUTPUT_STEM, conditions=CONDITIONS, labels=LABELS,
    )
    plt.close(fig)

    summary = summary_table(baseline, data)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    logger.info("=== Comparison complete ===")

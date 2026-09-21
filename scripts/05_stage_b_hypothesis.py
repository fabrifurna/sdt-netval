"""Stage B hypothesis testing — entry point.

Mann-Whitney U test of every experimental condition against the baseline, on the
key topological metrics, with Holm correction for multiple comparisons and
Cliff's delta as effect size. The statistics live in the library
(``sdt_netval.analysis.compare_to_baseline``); this script only wires the two
thesis datasets to it.

Usage:
    python scripts/05_stage_b_hypothesis.py legacy   # Tomašević: c0 vs c1..c10 (default)
    python scripts/05_stage_b_hypothesis.py cnr      # CNR: Stage A baseline vs 4 RecSys
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sdt_netval.analysis import compare_to_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "01_processed"

# Per-dataset wiring. `baseline_csv=None` means the baseline is the condition
# `baseline_condition` inside the Stage B table (legacy layout).
DATASETS = {
    "legacy": {
        "stage_b_csv": PROCESSED / "01_legacy_tomasevic" / "stage_b_raw.csv",
        "baseline_csv": None,
        "baseline_condition": "c0",
        # the four conditions analysed in the thesis (None = every condition)
        "conditions": ["c1", "c3", "c4", "c8"],
        "output_csv": PROCESSED / "01_legacy_tomasevic" / "stage_b_pvalues.csv",
    },
    "cnr": {
        "stage_b_csv": PROCESSED / "03_cnr_recsys_stageB" / "stage_b_raw.csv",
        "baseline_csv": PROCESSED / "02_cnr_baseline_stageA" / "stage_a_raw.csv",
        "baseline_condition": None,
        "conditions": None,
        "output_csv": PROCESSED / "03_cnr_recsys_stageB" / "stage_b_pvalues.csv",
    },
}
METRICS = ["alpha_in_degree", "modularity", "average_clustering"]


def run(dataset: str, correction: str = "holm") -> pd.DataFrame:
    cfg = DATASETS[dataset]
    data = pd.read_csv(cfg["stage_b_csv"])
    baseline = cfg["baseline_condition"] or pd.read_csv(cfg["baseline_csv"])

    results = compare_to_baseline(
        data, baseline, metrics=METRICS, conditions=cfg["conditions"], correction=correction,
    )

    cfg["output_csv"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg["output_csv"], index=False)
    logger.info("Test results saved — file: '%s', rows: %d", cfg["output_csv"], len(results))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", choices=DATASETS, default="legacy")
    parser.add_argument(
        "--correction", choices=["holm", "bonferroni", "none"], default="holm",
        help="Multiple-comparison correction (raw p-values are always kept in 'p_value').",
    )
    args = parser.parse_args()

    logger.info("=== Stage B — Hypothesis Testing (%s) ===", args.dataset)
    results = run(args.dataset, args.correction)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results.round(4).to_string(index=False))
    significant = results[results["significance"] != "ns"]
    logger.info(
        "Significant (%s correction): %d/%d tests", args.correction, len(significant), len(results)
    )

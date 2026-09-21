import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sdt_netval import StageAValidator, StageBAnalyzer, save_report_json
from sdt_netval.viz import plot_baseline_comparison

# ---------------------------------------------------------------- Stage A


def test_stage_a_processes_runs_and_aggregates(make_runs, tmp_path):
    d = make_runs("baseline", n_runs=4)
    v = StageAValidator(d)
    df = v.process_runs()
    assert len(df) == 4 and "error" not in df.columns
    assert set(df["run_id"]) == {"run0", "run1", "run2", "run3"}

    stab = v.compute_stability_metrics()
    assert stab.loc["modularity", "n"] == 4
    assert stab.loc["modularity", "mean"] == pytest.approx(df["modularity"].mean(), abs=1e-6)
    row = stab.loc["modularity"]
    assert row["ci95_low"] < row["mean"] < row["ci95_high"]

    out = tmp_path / "out" / "raw.csv"
    v.save_raw_results(out)
    assert len(pd.read_csv(out)) == 4


def test_stage_a_survives_a_corrupt_run(make_runs):
    d = make_runs("baseline", n_runs=2)
    (d / "broken.sqlite").write_text("this is not a database")
    df = StageAValidator(d).process_runs()
    assert len(df) == 3
    assert df["error"].notna().sum() == 1
    assert df.loc[df["error"].notna(), "run_id"].iloc[0] == "broken"


def test_stage_a_input_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        StageAValidator(tmp_path / "nope")
    with pytest.raises(ValueError, match="No supported"):
        StageAValidator(tmp_path)
    with pytest.raises(ValueError, match="process_runs"):
        (tmp_path / "a.csv").write_text("source,target\n1,2\n")
        StageAValidator(tmp_path).compute_stability_metrics()


def test_stage_a_reads_zip_archives(tmp_path, zip_with_db):
    d = tmp_path / "zips"
    d.mkdir()
    (d / "run_1.zip").write_bytes(zip_with_db.read_bytes())
    df = StageAValidator(d, tmp_dir=tmp_path / "scratch").process_runs()
    assert df["num_nodes"].iloc[0] == 4


# ---------------------------------------------------------------- Stage B


@pytest.fixture
def conditions_dir(make_runs, tmp_path):
    # different attachment parameter m -> genuinely different topologies per condition
    make_runs("cond/c1_low", n_runs=3, m=2)
    make_runs("cond/c2_high", n_runs=3, m=6, seed0=50)
    return tmp_path / "cond"


def test_stage_b_nested_pipeline(conditions_dir):
    a = StageBAnalyzer(conditions_dir)
    assert a.conditions == ["c1_low", "c2_high"]
    raw = a.process_all_runs()
    assert len(raw) == 6 and set(raw["condition"]) == {"c1_low", "c2_high"}

    agg = a.compute_aggregated_metrics()
    assert agg["condition"].tolist() == ["c1_low", "c2_high"]
    assert agg.set_index("condition").loc["c2_high", "density_mean"] > \
        agg.set_index("condition").loc["c1_low", "density_mean"]
    assert {"modularity_mean", "modularity_ci95_low", "modularity_ci95_high"} <= set(agg.columns)


def test_stage_b_explicit_conditions_and_errors(conditions_dir):
    (conditions_dir / "notes").mkdir()
    (conditions_dir / "baseline_like").mkdir()
    assert StageBAnalyzer(conditions_dir).conditions == ["c1_low", "c2_high"]  # 'notes' ignored
    picked = StageBAnalyzer(conditions_dir, conditions=["c2_high", "c1_low"])
    assert picked.conditions == ["c2_high", "c1_low"]
    with pytest.raises(ValueError, match="not found"):
        StageBAnalyzer(conditions_dir, conditions=["c9_ghost"])
    with pytest.raises(FileNotFoundError):
        StageBAnalyzer(conditions_dir / "nope")


def test_stage_b_report_requires_processing(conditions_dir):
    with pytest.raises(ValueError, match="process_all_runs"):
        StageBAnalyzer(conditions_dir).full_sensitivity_report()


# ---------------------------------------------------------------- export


def test_save_report_json_handles_frames_numpy_and_nan(tmp_path):
    report = {
        "raw": pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}),
        "scalar": np.float64(1.5),
        "int": np.int64(3),
        "bad": float("inf"),
        "nested": {"x": [np.float32(2.0)]},
    }
    out = tmp_path / "deep" / "dir" / "report.json"
    save_report_json(report, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["raw"] == [{"a": 1, "b": 0.5}, {"a": 2, "b": None}]
    assert loaded["scalar"] == 1.5 and loaded["int"] == 3
    assert loaded["bad"] is None
    assert not math.isnan(loaded["nested"]["x"][0])


# ---------------------------------------------------------------- viz


def test_plot_baseline_comparison_writes_png_and_pdf(runs_table, tmp_path):
    base = runs_table[runs_table["condition"] == "base"]
    data = runs_table[runs_table["condition"] != "base"]
    fig = plot_baseline_comparison(
        base, data, tmp_path / "figs" / "cmp.v1",   # dot in the stem must survive
        labels={"shifted": "Shifted"},
    )
    assert len(fig.axes) == 3
    plt.close(fig)
    for ext in ("png", "pdf"):
        f = tmp_path / "figs" / f"cmp.v1.{ext}"
        assert f.exists() and f.stat().st_size > 1000


def test_plot_custom_metrics_without_output(runs_table):
    base = runs_table[runs_table["condition"] == "base"]
    data = runs_table[runs_table["condition"] != "base"]
    fig = plot_baseline_comparison(base, data, metrics={"density": "Density"})
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_validation(runs_table):
    base = runs_table[runs_table["condition"] == "base"]
    data = runs_table[runs_table["condition"] != "base"]
    with pytest.raises(ValueError, match="Metric"):
        plot_baseline_comparison(base, data, metrics={"nope": "Nope"})
    with pytest.raises(ValueError, match="No runs"):
        plot_baseline_comparison(base, data, conditions=["ghost"])
    with pytest.raises(ValueError, match="Column"):
        plot_baseline_comparison(base, data, condition_col="missing")

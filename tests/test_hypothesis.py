import numpy as np
import pandas as pd
import pytest

from sdt_netval.analysis import adjust_pvalues, cliffs_delta, compare_to_baseline


def test_holm_matches_hand_computed_example():
    # sorted p: .01,.02,.03,.04 -> (4*.01, 3*.02, 2*.03, 1*.04) = .04,.06,.06,.04 -> running max
    adj = adjust_pvalues([0.04, 0.01, 0.03, 0.02], "holm")
    assert adj == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_bonferroni_none_and_cap():
    assert adjust_pvalues([0.4, 0.01], "bonferroni") == pytest.approx([0.8, 0.02])
    assert adjust_pvalues([0.9, 0.9], "bonferroni") == pytest.approx([1.0, 1.0])
    assert adjust_pvalues([0.4, 0.01], "none") == pytest.approx([0.4, 0.01])
    assert len(adjust_pvalues([], "holm")) == 0


def test_unknown_correction_raises():
    with pytest.raises(ValueError):
        adjust_pvalues([0.1], "fdr-magic")


def test_cliffs_delta_extremes_and_sign():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_detects_shift_and_ignores_null(runs_table):
    res = compare_to_baseline(runs_table, "base", metrics=["modularity"])
    by_cond = res.set_index("condition")
    assert by_cond.loc["shifted", "significance"] == "***"
    assert by_cond.loc["shifted", "cliffs_delta"] == pytest.approx(1.0)
    assert by_cond.loc["shifted", "effect_magnitude"] == "large"
    assert by_cond.loc["shifted", "pct_change"] > 50
    assert by_cond.loc["same", "significance"] == "ns"
    assert set(res["condition"]) == {"shifted", "same"}   # baseline excluded


def test_external_baseline_dataframe_equals_inline_baseline(runs_table):
    inline = compare_to_baseline(runs_table, "base", metrics=["alpha_in_degree"])
    base = runs_table[runs_table["condition"] == "base"]
    data = runs_table[runs_table["condition"] != "base"]
    external = compare_to_baseline(data, base, metrics=["alpha_in_degree"])
    pd.testing.assert_frame_equal(inline, external)


def test_correction_only_increases_pvalues_and_none_is_identity(runs_table):
    holm = compare_to_baseline(runs_table, "base", correction="holm")
    raw = compare_to_baseline(runs_table, "base", correction="none")
    assert (holm["p_adjusted"] >= holm["p_value"] - 1e-12).all()
    assert raw["p_adjusted"].tolist() == raw["p_value"].tolist()


def test_alternative_is_expressed_for_the_condition(runs_table):
    greater = compare_to_baseline(runs_table, "base", metrics=["modularity"],
                                  alternative="greater", correction="none")
    less = compare_to_baseline(runs_table, "base", metrics=["modularity"],
                               alternative="less", correction="none")
    p_g = greater.set_index("condition").loc["shifted", "p_value"]
    p_l = less.set_index("condition").loc["shifted", "p_value"]
    assert p_g < 0.001 and p_l > 0.99   # modularity of "shifted" is higher than baseline


def test_selects_conditions_and_output_order(runs_table):
    res = compare_to_baseline(
        runs_table, "base", metrics=["modularity"], conditions=["same", "shifted"]
    )
    assert res["condition"].tolist() == ["same", "shifted"]


def test_missing_metric_or_small_group_is_skipped_not_fatal(runs_table):
    res = compare_to_baseline(runs_table, "base", metrics=["modularity", "nonexistent"])
    assert set(res["metric"]) == {"modularity"}
    one_run = pd.concat([runs_table, runs_table[runs_table["condition"] == "same"].head(1)
                         .assign(condition="tiny")])
    tiny = compare_to_baseline(one_run, "base", metrics=["modularity"], conditions=["tiny", "same"])
    assert tiny["condition"].tolist() == ["same"]


def test_nan_runs_are_dropped(runs_table):
    runs_table.loc[runs_table.index[0], "modularity"] = np.nan
    res = compare_to_baseline(runs_table, "base", metrics=["modularity"])
    assert res["n_baseline"].iloc[0] == 11


@pytest.mark.parametrize("kwargs, exc", [
    ({"baseline": "missing"}, ValueError),
    ({"baseline": "base", "condition_col": "nope"}, ValueError),
    ({"baseline": "base", "alternative": "sideways"}, ValueError),
    ({"baseline": "base", "correction": "nah"}, ValueError),
])
def test_invalid_inputs_raise(runs_table, kwargs, exc):
    with pytest.raises(exc):
        compare_to_baseline(runs_table, **kwargs)

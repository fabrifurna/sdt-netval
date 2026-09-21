import math

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from sdt_netval import GraphMetrics
from sdt_netval.core.stats import mean_ci95


def test_rejects_non_digraph_and_empty_graph():
    with pytest.raises(TypeError):
        GraphMetrics(nx.Graph([(0, 1)]))
    with pytest.raises(ValueError):
        GraphMetrics(nx.DiGraph())


def test_basic_stats_on_complete_digraph():
    G = nx.complete_graph(5, create_using=nx.DiGraph)
    s = GraphMetrics(G).compute_basic_stats()
    assert s["num_nodes"] == 5 and s["num_edges"] == 20
    assert s["density"] == pytest.approx(1.0)
    assert s["reciprocity"] == pytest.approx(1.0)
    assert s["avg_in_degree"] == pytest.approx(4.0)
    assert s["std_in_degree"] == pytest.approx(0.0)


def test_reciprocity_of_a_directed_cycle_is_zero():
    G = nx.cycle_graph(6, create_using=nx.DiGraph)
    assert GraphMetrics(G).compute_basic_stats()["reciprocity"] == 0.0


def test_modularity_detects_two_communities():
    a = nx.complete_graph(range(0, 8))
    b = nx.complete_graph(range(8, 16))
    G = nx.compose(a, b)
    G.add_edge(0, 8)
    m = GraphMetrics(G.to_directed()).compute_clustering_and_modularity()
    assert m["num_communities"] == 2
    assert m["modularity"] > 0.4
    assert m["average_clustering"] > 0.9


def test_powerlaw_skipped_on_tiny_graph():
    out = GraphMetrics(nx.path_graph(5, create_using=nx.DiGraph)).compute_powerlaw_alpha()
    assert out["alpha_in_degree"] is None
    assert "Insufficient" in out["powerlaw_error"]


def test_powerlaw_on_scale_free_graph_gives_plausible_alpha():
    G = nx.DiGraph(nx.barabasi_albert_graph(400, 3, seed=1).to_directed())
    out = GraphMetrics(G).compute_powerlaw_alpha()
    assert out["powerlaw_error"] is None
    assert 1.5 < out["alpha_in_degree"] < 4.5


def test_full_report_is_flat_and_complete():
    G = nx.DiGraph(nx.barabasi_albert_graph(100, 3, seed=2).to_directed())
    report = GraphMetrics(G).generate_full_report()
    for key in ("num_nodes", "density", "alpha_in_degree", "modularity", "average_clustering"):
        assert key in report
    assert all(not isinstance(v, (dict, list)) for v in report.values())


def test_mean_ci95_matches_scipy():
    x = np.array([0.31, 0.29, 0.35, 0.30, 0.33, 0.28])
    out = mean_ci95(x)
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    assert out["mean"] == pytest.approx(x.mean())
    assert out["std"] == pytest.approx(x.std(ddof=1))
    assert (out["ci_low"], out["ci_high"]) == (pytest.approx(lo), pytest.approx(hi))


def test_mean_ci95_edge_cases():
    assert mean_ci95([1.0, float("nan"), 3.0])["n"] == 2
    single = mean_ci95([5.0])
    assert single["ci_low"] == single["ci_high"] == 5.0
    with pytest.raises(ValueError):
        mean_ci95([float("nan")])
    assert not math.isnan(mean_ci95([1, 2, 3])["sem"])

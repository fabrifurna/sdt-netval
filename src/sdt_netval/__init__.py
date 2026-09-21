"""sdt-netval: Universal topology validator for multi-agent simulation networks.

Quick start::

    from sdt_netval import load_network, GraphMetrics

    G = load_network("simulation.sqlite")   # or .zip or .csv
    report = GraphMetrics(G).generate_full_report()

Multi-run studies (baseline vs experimental conditions)::

    from sdt_netval import StageAValidator, StageBAnalyzer, compare_to_baseline

    baseline = StageAValidator("runs/baseline/")
    baseline.process_runs()
    stage_b = StageBAnalyzer("runs/conditions/")
    stage_b.process_all_runs()
    tests = compare_to_baseline(
        stage_b.full_sensitivity_report()["raw"], baseline.full_stability_report()["raw"]
    )

Figures live in ``sdt_netval.viz`` (imported on demand to keep this import light).
"""

from sdt_netval.adapters import load_network
from sdt_netval.analysis import compare_to_baseline
from sdt_netval.core.metrics import KEY_METRICS, GraphMetrics
from sdt_netval.export import save_report_json
from sdt_netval.pipeline import StageAValidator, StageBAnalyzer

__all__ = [
    "load_network",
    "GraphMetrics",
    "KEY_METRICS",
    "StageAValidator",
    "StageBAnalyzer",
    "compare_to_baseline",
    "save_report_json",
]
__version__ = "0.1.0"

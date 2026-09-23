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

Every name below is resolved lazily, on first access (PEP 562). A plain
``import sdt_netval``, or importing a submodule such as ``sdt_netval.cli`` (which
Python resolves by first running this file), does not pull in pandas, networkx,
scipy, powerlaw or matplotlib until something actually asks for them.
"""

import importlib
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "load_network",
    "GraphMetrics",
    "KEY_METRICS",
    "StageAValidator",
    "StageBAnalyzer",
    "compare_to_baseline",
    "save_report_json",
]

# name -> submodule that defines it
_LAZY_ATTRS = {
    "load_network": "sdt_netval.adapters",
    "GraphMetrics": "sdt_netval.core.metrics",
    "KEY_METRICS": "sdt_netval.core.metrics",
    "StageAValidator": "sdt_netval.pipeline",
    "StageBAnalyzer": "sdt_netval.pipeline",
    "compare_to_baseline": "sdt_netval.analysis",
    "save_report_json": "sdt_netval.export",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: the import only happens on first access
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_ATTRS))

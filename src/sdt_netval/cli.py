"""Command-line interface for sdt-netval.

Usage examples:
    sdt-netval analyze simulation.sqlite
    sdt-netval analyze edges.csv --output results/
    sdt-netval stage-a data/benchmark_runs/
    sdt-netval stage-a data/benchmark_runs/ --output results/ --format json
    sdt-netval stage-b data/sensitivity_runs/ --output results/
    sdt-netval compare stage_a_raw.csv stage_b_raw.csv --output results/ --plot
    sdt-netval compare stage_b_raw.csv --baseline-condition c0
"""

import argparse
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence


def _package_version() -> str:
    # Reads the version from the installed package's metadata instead of doing
    # `from sdt_netval import __version__`, which would import the whole package
    # (and with it pandas, networkx, scipy, powerlaw, and matplotlib via powerlaw)
    # just to answer `--version` or print `--help`.
    try:
        return metadata.version("sdt-netval")
    except metadata.PackageNotFoundError:
        # Not installed (e.g. running from a source checkout without `pip install -e .`)
        from sdt_netval import __version__

        return __version__


def _make_streams_tolerant() -> None:
    # A redirected Windows console (or one still on cp1252) raises UnicodeEncodeError
    # on the first non-ASCII character, and our log messages contain a few.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> int:
    from sdt_netval import GraphMetrics, load_network
    from sdt_netval.export import save_report_json

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: '{path}'", file=sys.stderr)
        return 1


    print(f"\nLoading '{path.name}' ...")
    try:
        G = load_network(path, tmp_dir=args.tmp_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"  nodes: {G.number_of_nodes():,}   edges: {G.number_of_edges():,}\n")

    report = GraphMetrics(G).generate_full_report()

    _print_metrics_table(report)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        save_report_json(report, out_dir / f"{stem}_metrics.json")
        print(f"\nResults saved to '{out_dir}/'")

    return 0


def _print_metrics_table(report: dict) -> None:
    sections = {
        "Network structure": [
            ("num_nodes",     "Nodes"),
            ("num_edges",     "Edges"),
            ("density",       "Density"),
            ("avg_in_degree", "Avg in-degree"),
            ("std_in_degree", "Std in-degree"),
            ("reciprocity",   "Reciprocity"),
        ],
        "Degree distribution (power-law)": [
            ("alpha_in_degree", "Alpha"),
            ("xmin",            "x_min"),
            ("ks_distance",     "KS distance"),
            ("is_scale_free",   "Scale-free (2 < alpha < 3)"),
            ("powerlaw_error",  "Fit error"),
        ],
        "Community structure": [
            ("average_clustering", "Avg clustering"),
            ("modularity",         "Modularity (Q)"),
            ("num_communities",    "Communities"),
        ],
    }

    for section, fields in sections.items():
        print(f"  {section}")
        print(f"  {'-' * 42}")
        for key, label in fields:
            val = report.get(key)
            if val is None:
                formatted = "n/a"
            elif isinstance(val, bool):
                formatted = "yes" if val else "no"
            elif isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            print(f"  {label:<28} {formatted}")
        print()


# ---------------------------------------------------------------------------
# stage-a
# ---------------------------------------------------------------------------

def _cmd_stage_a(args: argparse.Namespace) -> int:
    from sdt_netval.export import save_report_json
    from sdt_netval.pipeline import StageAValidator

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: directory not found: '{data_dir}'", file=sys.stderr)
        return 1


    try:
        validator = StageAValidator(data_dir, tmp_dir=args.tmp_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nStage A: {len(validator.db_paths)} run(s) found in '{data_dir.name}'\n")
    validator.process_runs()

    report = validator.full_stability_report()

    print("=== Stability Statistics ===\n")
    print(report["stability"].to_string())
    print()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report["raw"].to_csv(out_dir / "stage_a_raw.csv", index=False)
        report["stability"].to_csv(out_dir / "stage_a_stability.csv")

        if args.format == "json":
            save_report_json(report, out_dir / "stage_a_report.json")

        print(f"Results saved to '{out_dir}/'")

    return 0


# ---------------------------------------------------------------------------
# stage-b
# ---------------------------------------------------------------------------

def _cmd_stage_b(args: argparse.Namespace) -> int:
    from sdt_netval.export import save_report_json
    from sdt_netval.pipeline import StageBAnalyzer

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: directory not found: '{base_dir}'", file=sys.stderr)
        return 1


    try:
        analyzer = StageBAnalyzer(base_dir, conditions=args.conditions, tmp_dir=args.tmp_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nStage B: {len(analyzer.conditions)} condition(s): {analyzer.conditions}\n")
    analyzer.process_all_runs()

    report = analyzer.full_sensitivity_report()

    print("=== Per-condition Aggregated Metrics ===\n")
    print(report["aggregated"].to_string(index=False))
    print()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report["raw"].to_csv(out_dir / "stage_b_raw.csv", index=False)
        report["aggregated"].to_csv(out_dir / "stage_b_aggregated.csv", index=False)

        if args.format == "json":
            save_report_json(report, out_dir / "stage_b_report.json")

        print(f"Results saved to '{out_dir}/'")

    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    from sdt_netval.analysis import compare_to_baseline

    if (args.conditions_csv is None) == (args.baseline_condition is None):
        print(
            "Error: give either two CSVs (baseline + conditions) or one CSV with "
            "--baseline-condition.",
            file=sys.stderr,
        )
        return 1

    try:
        first = pd.read_csv(args.baseline)
        if args.conditions_csv is not None:
            data, baseline = pd.read_csv(args.conditions_csv), first
        else:
            data, baseline = first, args.baseline_condition
        results = compare_to_baseline(
            data, baseline,
            metrics=args.metrics,
            condition_col=args.condition_col,
            correction=args.correction,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== Mann-Whitney U vs baseline ({args.correction} correction) ===\n")
    print(results.round(4).to_string(index=False))
    print("\nSignificance is computed on the adjusted p-value: *** <0.001, ** <0.01, * <0.05.")
    print("Cliff's delta > 0: the condition tends to exceed the baseline.\n")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_dir / "comparison_tests.csv", index=False)

        if args.plot:
            import matplotlib.pyplot as plt

            from sdt_netval.viz import plot_baseline_comparison

            base_df = baseline if args.conditions_csv is not None else data[
                data[args.condition_col] == baseline]
            cond_df = data if args.conditions_csv is not None else data[
                data[args.condition_col] != baseline]
            fig = plot_baseline_comparison(
                base_df, cond_df, out_dir / "baseline_comparison",
                condition_col=args.condition_col,
            )
            plt.close(fig)
        print(f"Results saved to '{out_dir}/'")
    elif args.plot:
        print("Note: --plot needs --output DIR; no figure written.", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sdt-netval",
        description="Universal topology validator for multi-agent simulation networks.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    tmp_help = (
        "Directory used to extract databases from .zip archives (they can be several GB). "
        "Defaults to $SDT_TMPDIR, then the system temp dir."
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- analyze --
    p_analyze = sub.add_parser(
        "analyze",
        help="Compute topology metrics for a single network file (.sqlite/.db, .csv, .zip).",
    )
    p_analyze.add_argument("file", help="Path to the network file.")
    p_analyze.add_argument(
        "--output", "-o", metavar="DIR",
        help="Directory where results are saved (JSON). Omit to only print to screen.",
    )
    p_analyze.add_argument("--tmp-dir", metavar="DIR", help=tmp_help)
    p_analyze.set_defaults(func=_cmd_analyze)

    # -- stage-a --
    p_a = sub.add_parser(
        "stage-a",
        help="Run Stage A stability pipeline on a directory of simulation files.",
    )
    p_a.add_argument("data_dir", help="Directory containing simulation files.")
    p_a.add_argument("--output", "-o", metavar="DIR", help="Directory for output CSVs.")
    p_a.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="Output format (default: csv).",
    )
    p_a.add_argument("--tmp-dir", metavar="DIR", help=tmp_help)
    p_a.set_defaults(func=_cmd_stage_a)

    # -- stage-b --
    p_b = sub.add_parser(
        "stage-b",
        help="Run Stage B sensitivity pipeline on a nested conditions directory.",
    )
    p_b.add_argument(
        "base_dir", help="Directory containing condition subdirectories (c0, c1, ...)."
    )
    p_b.add_argument("--output", "-o", metavar="DIR", help="Directory for output CSVs.")
    p_b.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="Output format (default: csv).",
    )
    p_b.add_argument(
        "--conditions", nargs="+", metavar="NAME",
        help="Condition subdirectories to use, in order (default: auto-discover 'c*').",
    )
    p_b.add_argument("--tmp-dir", metavar="DIR", help=tmp_help)
    p_b.set_defaults(func=_cmd_stage_b)

    # -- compare --
    p_c = sub.add_parser(
        "compare",
        help="Mann-Whitney U tests of each condition against a baseline (from raw CSVs).",
    )
    p_c.add_argument("baseline", help="Baseline raw CSV (e.g. stage_a_raw.csv), or the only CSV.")
    p_c.add_argument(
        "conditions_csv", nargs="?", help="Conditions raw CSV (e.g. stage_b_raw.csv).",
    )
    p_c.add_argument(
        "--baseline-condition", metavar="NAME",
        help="Use this condition of the single CSV as baseline (e.g. c0).",
    )
    p_c.add_argument("--condition-col", default="condition", help="Condition column name.")
    p_c.add_argument(
        "--metrics", nargs="+", metavar="M", help="Metrics to test (default: key metrics)."
    )
    p_c.add_argument(
        "--correction", choices=["holm", "bonferroni", "none"], default="holm",
        help="Multiple-comparison correction across conditions (default: holm).",
    )
    p_c.add_argument("--output", "-o", metavar="DIR", help="Directory for the results CSV.")
    p_c.add_argument(
        "--plot", action="store_true", help="Also save the comparison figure (needs -o)."
    )
    p_c.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    _make_streams_tolerant()
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

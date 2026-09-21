import pytest

import sdt_netval
from sdt_netval.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert sdt_netval.__version__ in capsys.readouterr().out


def test_analyze_prints_metrics_and_saves_json(make_runs, tmp_path, capsys):
    db = next(make_runs("one", n_runs=1).glob("*.sqlite"))
    out = tmp_path / "res"
    assert main(["analyze", str(db), "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Modularity (Q)" in text and "Nodes" in text
    assert (out / f"{db.stem}_metrics.json").exists()


def test_analyze_reports_errors_with_nonzero_exit(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "ghost.sqlite")]) == 1
    bad = tmp_path / "x.xlsx"
    bad.write_text("x")
    assert main(["analyze", str(bad)]) == 1
    assert "Error" in capsys.readouterr().err


def test_stage_a_and_b_write_outputs(make_runs, tmp_path):
    a_dir = make_runs("base", n_runs=3)
    out_a = tmp_path / "out_a"
    assert main(["stage-a", str(a_dir), "-o", str(out_a), "--format", "json"]) == 0
    assert {"stage_a_raw.csv", "stage_a_stability.csv", "stage_a_report.json"} <= {
        p.name for p in out_a.iterdir()}

    make_runs("cond/c1_x", n_runs=2, m=2)
    make_runs("cond/c2_y", n_runs=2, m=5, seed0=9)
    out_b = tmp_path / "out_b"
    assert main(["stage-b", str(tmp_path / "cond"), "-o", str(out_b),
                 "--conditions", "c2_y", "c1_x"]) == 0
    agg = (out_b / "stage_b_aggregated.csv").read_text()
    assert agg.index("c2_y") < agg.index("c1_x")   # explicit order respected


def test_compare_from_two_csvs_and_plot(make_runs, tmp_path, capsys):
    base_dir = make_runs("base", n_runs=4, m=2)
    make_runs("cond/c1_dense", n_runs=4, m=8, seed0=20)
    out = tmp_path / "cmp"
    assert main(["stage-a", str(base_dir), "-o", str(tmp_path / "a")]) == 0
    assert main(["stage-b", str(tmp_path / "cond"), "-o", str(tmp_path / "b")]) == 0

    code = main([
        "compare", str(tmp_path / "a" / "stage_a_raw.csv"), str(tmp_path / "b" / "stage_b_raw.csv"),
        "-o", str(out), "--plot",
    ])
    assert code == 0
    assert "Mann-Whitney" in capsys.readouterr().out
    assert (out / "comparison_tests.csv").exists()
    assert (out / "baseline_comparison.png").exists()
    assert (out / "baseline_comparison.pdf").exists()


def test_compare_single_csv_with_baseline_condition(make_runs, tmp_path):
    make_runs("cond/c0", n_runs=4, m=2)
    make_runs("cond/c1", n_runs=4, m=8, seed0=20)
    assert main(["stage-b", str(tmp_path / "cond"), "-o", str(tmp_path / "b")]) == 0
    raw = str(tmp_path / "b" / "stage_b_raw.csv")
    assert main(["compare", raw, "--baseline-condition", "c0", "--correction", "bonferroni"]) == 0


def test_compare_argument_errors(tmp_path, capsys):
    csv = tmp_path / "r.csv"
    csv.write_text("condition,modularity\nc0,0.1\n")
    assert main(["compare", str(csv)]) == 1                      # neither mode given
    assert main(["compare", str(csv), str(csv), "--baseline-condition", "c0"]) == 1   # both given
    assert main(["compare", str(csv), "--baseline-condition", "zzz"]) == 1  # unknown baseline
    assert "Error" in capsys.readouterr().err


def _run_cp1252(args, env_extra=None):
    """Run a python subprocess whose stdout/stderr are cp1252, like a redirected Windows console."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    env["PYTHONPATH"] = str(Path(sdt_netval.__file__).resolve().parents[1])
    return subprocess.run([sys.executable, *args], capture_output=True, env=env, timeout=120)


def test_analyze_output_is_plain_ascii(make_runs):
    # Used to die with UnicodeEncodeError on redirected Windows output (box-drawing chars, alpha).
    # A '?' in the output would mean a non-ASCII character was printed and got replaced.
    db = next(make_runs("enc", n_runs=1).glob("*.sqlite"))
    proc = _run_cp1252(["-m", "sdt_netval.cli", "analyze", str(db)])
    assert proc.returncode == 0, proc.stderr.decode("cp1252", errors="replace")
    assert b"Modularity" in proc.stdout
    assert b"?" not in proc.stdout


def test_stream_safety_net_replaces_instead_of_crashing():
    code = (
        "from sdt_netval.cli import _make_streams_tolerant; _make_streams_tolerant(); "
        "print('alpha=\u03b1')"
    )
    proc = _run_cp1252(["-c", code])
    assert proc.returncode == 0, proc.stderr.decode("cp1252", errors="replace")
    assert proc.stdout.strip() == b"alpha=?"

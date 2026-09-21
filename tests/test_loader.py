import tempfile
from pathlib import Path

import pytest
from conftest import make_zip, write_sqlite

from sdt_netval import load_network


def test_sqlite_resolves_follow_unfollow(small_db):
    G = load_network(small_db)
    assert set(G.nodes) == {0, 1, 2, 3}
    assert set(G.edges) == {(0, 1), (1, 2), (2, 3), (3, 0)}  # (0,2) unfollowed, (3,0) restored
    assert G.edges[0, 1]["round_created"] == 1


def test_sqlite_end_round_truncates_history(small_db):
    G = load_network(small_db, end_round=2)
    # at round 2 the unfollow (round 3) has not happened yet, (3,0) does not exist yet
    assert set(G.edges) == {(0, 1), (1, 2), (2, 3), (0, 2)}


def test_sqlite_empty_user_table_raises(tmp_path):
    db = write_sqlite(tmp_path / "empty.sqlite", 0, [])
    with pytest.raises(ValueError, match="empty"):
        load_network(db)


def test_csv_default_columns_keep_edge_attributes(edge_csv):
    G = load_network(edge_csv)
    assert G.number_of_edges() == 3
    assert G.edges[1, 2]["weight"] == 2


def test_csv_autodetects_alias_columns(tmp_path):
    path = tmp_path / "aliases.csv"
    path.write_text("follower_id,user_id\n1,2\n2,3\n")
    G = load_network(path)
    assert set(G.edges) == {(1, 2), (2, 3)}


def test_csv_unknown_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="auto-detect|not found"):
        load_network(path)


def test_csv_explicit_columns(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("a,b\n1,2\n2,3\n")
    G = load_network(path, source_col="a", target_col="b")
    assert set(G.edges) == {(1, 2), (2, 3)}


def test_zip_with_sqlite_matches_direct_load(zip_with_db, small_db):
    assert set(load_network(zip_with_db).edges) == set(load_network(small_db).edges)


def test_zip_with_csv(tmp_path, edge_csv):
    z = make_zip(tmp_path / "csv.zip", {"network_edges.csv": edge_csv.read_text()})
    assert load_network(z).number_of_edges() == 3


def test_zip_edge_file_override(tmp_path, edge_csv, small_db):
    z = make_zip(
        tmp_path / "both.zip",
        {"other.csv": "source,target\n9,8\n", "edges.csv": edge_csv.read_text()},
    )
    G = load_network(z, edge_file="edges.csv")
    assert set(G.nodes) == {0, 1, 2}


def test_zip_json_only_not_implemented(tmp_path):
    z = make_zip(tmp_path / "json.zip", {"network.json": "{}"})
    with pytest.raises(NotImplementedError):
        load_network(z)


def test_zip_without_network_data_raises(tmp_path):
    z = make_zip(tmp_path / "junk.zip", {"readme.txt": "x"})
    with pytest.raises(ValueError, match="No parseable network data"):
        load_network(z)


@pytest.mark.parametrize("via_env", [False, True])
def test_zip_extracts_in_requested_dir_and_cleans_up(zip_with_db, tmp_path, monkeypatch, via_env):
    """The multi-GB database must be extracted where the user asked, then removed."""
    scratch = tmp_path / "scratch"
    seen = []
    real = tempfile.TemporaryDirectory

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", spy)
    if via_env:
        monkeypatch.setenv("SDT_TMPDIR", str(scratch))
        G = load_network(zip_with_db)
    else:
        G = load_network(zip_with_db, tmp_dir=scratch)

    assert G.number_of_edges() == 4
    assert Path(seen[0]) == scratch.resolve()
    assert list(scratch.iterdir()) == []  # volatile directory removed


def test_missing_file_and_unsupported_extension(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_network(tmp_path / "nope.sqlite")
    bad = tmp_path / "data.xlsx"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_network(bad)

"""Shared fixtures: synthetic simulation exports built on the fly (no real data needed)."""

import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting in CI


def write_sqlite(
    path: Path,
    n_nodes: int,
    follows: Iterable[Tuple[int, int, str, int]],
) -> Path:
    """Create a minimal Y-Social-like database: user_mgmt + follow tables."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE user_mgmt (id INTEGER PRIMARY KEY, username TEXT)")
        conn.execute(
            "CREATE TABLE follow (follower_id INTEGER, user_id INTEGER, action TEXT, round INTEGER)"
        )
        conn.executemany(
            "INSERT INTO user_mgmt VALUES (?, ?)", [(i, f"u{i}") for i in range(n_nodes)]
        )
        conn.executemany("INSERT INTO follow VALUES (?, ?, ?, ?)", list(follows))
    conn.close()
    return path


def random_follows(n_nodes: int, seed: int, m: int = 3):
    """Follow events of a scale-free-ish directed graph (Barabási–Albert edges)."""
    g = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    # newer node (v) follows older node (u): in-degree is heavy-tailed like a real follower graph
    return [(v, u, "follow", 1 + (u + v) % 5) for u, v in g.edges()]


@pytest.fixture
def small_db(tmp_path) -> Path:
    """4 nodes; includes an unfollow that must remove an edge, and a re-follow."""
    follows = [
        (0, 1, "follow", 1),
        (1, 2, "follow", 1),
        (2, 3, "follow", 2),
        (0, 2, "follow", 2),
        (0, 2, "unfollow", 3),   # edge 0->2 is removed
        (3, 0, "follow", 3),
        (3, 0, "unfollow", 4),
        (3, 0, "follow", 5),     # ... and restored
    ]
    return write_sqlite(tmp_path / "small.sqlite", 4, follows)


@pytest.fixture
def edge_csv(tmp_path) -> Path:
    path = tmp_path / "edges.csv"
    pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0], "weight": [1, 2, 3]}).to_csv(
        path, index=False
    )
    return path


@pytest.fixture
def make_runs(tmp_path):
    """Factory: directory with `n_runs` synthetic SQLite simulations."""

    def _make(name: str, n_runs: int = 3, n_nodes: int = 60, seed0: int = 0, m: int = 3) -> Path:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_runs):
            write_sqlite(
                d / f"run{i}.sqlite", n_nodes, random_follows(n_nodes, seed0 + i, m=m)
            )
        return d

    return _make


def make_zip(path: Path, members: dict) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return path


@pytest.fixture
def zip_with_db(tmp_path, small_db) -> Path:
    return make_zip(
        tmp_path / "export.zip",
        {"database_server.db": small_db.read_bytes(), "logs/server.log": "irrelevant"},
    )


@pytest.fixture
def runs_table():
    """Per-run metric table: baseline + 2 conditions, 12 runs each, clearly separated."""
    import numpy as np

    rng = np.random.default_rng(0)

    def block(name, q, alpha, clustering, n=12):
        return pd.DataFrame({
            "condition": name,
            "modularity": rng.normal(q, 0.005, n),
            "alpha_in_degree": rng.normal(alpha, 0.05, n),
            "average_clustering": rng.normal(clustering, 0.02, n),
            "density": rng.normal(0.03, 0.002, n),
        })

    return pd.concat(
        [block("base", 0.17, 2.6, 0.50), block("shifted", 0.32, 2.2, 0.50),
         block("same", 0.17, 2.6, 0.50)],
        ignore_index=True,
    )

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

First public release.

### Added
- `load_network()`: universal loader for `.sqlite`/`.db`, `.zip` and `.csv` simulation exports,
  with follow/unfollow resolution and edge-list column auto-detection.
- `GraphMetrics`: density, degree statistics, reciprocity, power-law exponent α of the in-degree
  distribution, average clustering and modularity Q.
- `StageAValidator` (stability of N baseline runs) and `StageBAnalyzer` (N conditions × M runs),
  with 95% t-distribution confidence intervals.
- `compare_to_baseline()`: Mann-Whitney U tests of each condition against a baseline, with
  Holm/Bonferroni correction and Cliff's delta effect size.
- `sdt_netval.viz.plot_baseline_comparison()`: publication-ready PNG/PDF comparison figure.
- `sdt-netval` command line: `analyze`, `stage-a`, `stage-b`, `compare`.
- `tmp_dir` / `--tmp-dir` / `SDT_TMPDIR`: choose where SQLite databases are extracted from ZIP
  archives (they can be several GB).
- `StageBAnalyzer(conditions=[...])` to select condition directories explicitly.

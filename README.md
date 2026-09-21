# sdt-netval

Topology checks for social simulations driven by LLM agents.

The idea behind it: a language model can write perfectly believable posts and still
end up producing a network that looks nothing like a real one. If you only judge a
simulation by how its text reads, you miss that. `sdt-netval` takes the follower
network a simulation produced and measures the things network scientists usually
look at (degree distribution, community structure, clustering). It can also compare
several experimental conditions, for example different recommender systems, against a
baseline and tell you whether the differences are more than run-to-run noise.

It was developed first of all for [Y Social](https://arxiv.org/abs/2408.00818), the
LLM-powered social media digital twin built by Giulio Rossetti's group at CNR-ISTI
(Rossetti et al., 2024), as part of a master's thesis at the University of Pisa
supervised by Prof. Rossetti. The SQLite reader expects the Y Social database layout,
and all the thesis experiments are Y Social runs: baseline and recommender-system
runs from the CNR group, plus the Voat-like forum runs of Tomašević et al. The
analysis itself is not tied to Y Social: anything that gives you a directed edge list
will do.

## Installation

**Not on PyPI yet.** The first release is still to come, so for now install from GitHub:

```bash
pip install git+https://github.com/fabrifurna/sdt-netval.git
```

Once it is published this becomes `pip install sdt-netval`.

Python 3.10 or newer. Installing also gives you the `sdt-netval` command.

## A first look

Point it at one network file:

```bash
sdt-netval analyze network.csv
```

```text
Loading 'network.csv' ...
  nodes: 500   edges: 1,491

  Network structure
  ------------------------------------------
  Nodes                        500
  Edges                        1491
  Density                      0.0060
  Avg in-degree                2.9820
  Std in-degree                6.1424
  Reciprocity                  0.0000

  Degree distribution (power-law)
  ------------------------------------------
  Alpha                        2.5019
  x_min                        6.0000
  KS distance                  0.0478
  Scale-free (2 < alpha < 3)   yes
  Fit error                    n/a

  Community structure
  ------------------------------------------
  Avg clustering               0.0467
  Modularity (Q)               0.3825
  Communities                  12
```

(That is a synthetic Barabási–Albert graph, so it is a nice power law and not much else.)

The same thing from Python:

```python
from sdt_netval import load_network, GraphMetrics

G = load_network("network.csv")          # .sqlite, .zip or .csv, picked from the extension
report = GraphMetrics(G).generate_full_report()
report["modularity"]
```

`report` is a flat dictionary, so a list of them drops straight into a DataFrame.

## What it reads

| Input | What it expects |
| --- | --- |
| `.csv` | One row per edge. Column names are guessed (`source`/`target`, `follower`/`followee`, `from`/`to`, ...). Pass `source_col=` and `target_col=` if yours are unusual. Extra columns become edge attributes. |
| `.sqlite`, `.db` | A Y-Social style database: a `user_mgmt` table with an `id` column, and a `follow` table with `follower_id`, `user_id`, `action` and `round`. Follow and unfollow events are replayed, so only the edges still standing at the end (or at `end_round=`) are kept. |
| `.zip` | An archive holding one of the above. Only the database is extracted; logs and configs are left alone. |

If your simulator uses a different database layout, export the edges to CSV. JSON
network files are not supported yet.

ZIP exports from long simulations can hold databases of several GB. They are unpacked
to a temporary directory that is deleted afterwards, and you can choose where it goes,
which helps when the system drive is small:

```bash
sdt-netval stage-a runs/ --tmp-dir D:\scratch
```

or set the `SDT_TMPDIR` environment variable, or pass `tmp_dir=` to `load_network`.

## What gets measured

- **Basic structure**: nodes, edges, density, mean and standard deviation of in- and
  out-degree, reciprocity.
- **α**: exponent of a discrete power-law fit to the in-degree distribution (using the
  [`powerlaw`](https://github.com/jeffalstott/powerlaw) package, nodes with in-degree 0
  left out, at least 10 needed). `is_scale_free` is just `2 < α < 3`. That is a rule of
  thumb, not a goodness-of-fit test.
- **Average clustering** and **modularity Q**, both on the undirected version of the graph.
  Communities come from greedy modularity maximisation, which is deterministic but is
  only one of several possible partitions.

## Running many simulations

A single run tells you little because simulations are stochastic. The intended workflow
has two stages.

Stage A: several runs of the baseline, to see how much the metrics move on their own.

```bash
sdt-netval stage-a baseline_runs/ -o results/baseline
```

Stage B: one folder per experimental condition, each with several runs.

```text
conditions/
├── popularity/
│   ├── run_1.zip
│   └── ...
└── follower/
    └── ...
```

```bash
sdt-netval stage-b conditions/ -o results/conditions
```

Both write the per-run table (`*_raw.csv`) and a summary with mean, standard deviation
and a 95% confidence interval (t-distribution, since N is usually 10 to 30). By default
every subfolder whose name starts with `c` or `condition_` is treated as a condition;
use `--conditions a b c` to choose folders and their order.

## Comparing against a baseline

```bash
sdt-netval compare results/baseline/stage_a_raw.csv results/conditions/stage_b_raw.csv -o results/ --plot
```

For each metric and each condition this runs a two-sided Mann-Whitney U test against
the baseline runs, corrects the p-values with Holm's method (across the conditions,
separately for each metric), and reports Cliff's delta as an effect size next to the
p-value. With samples this small a p-value alone is easy to over-read, which is why the
effect size is there. `--plot` also saves a figure as PNG and PDF, ready for a paper.

In Python:

```python
import pandas as pd
from sdt_netval import compare_to_baseline
from sdt_netval.viz import plot_baseline_comparison

baseline = pd.read_csv("results/baseline/stage_a_raw.csv")
runs = pd.read_csv("results/conditions/stage_b_raw.csv")

tests = compare_to_baseline(runs, baseline)
fig = plot_baseline_comparison(baseline, runs, "figures/comparison")
```

The columns of `tests` are `metric`, `condition`, `n_baseline`, `n_condition`,
`mean_baseline`, `mean_condition`, `mean_diff`, `pct_change`, `U`, `p_value`,
`p_adjusted`, `significance` (based on `p_adjusted`), `cliffs_delta` (positive when the
condition tends to be higher than the baseline) and `effect_magnitude`.

If baseline and conditions live in one table, pass the name of the baseline condition
instead of a DataFrame: `compare_to_baseline(runs, "c0")`.

## Things to keep in mind

- It only looks at network structure. It says nothing about what the agents write.
- The graph is treated as a simple directed graph: repeated edges collapse into one.
- Every user in the database is a node, including those who never follow anyone. In a
  very sparse simulation most nodes are isolated and each isolated node counts as its own
  community, so `num_communities` and modularity say little. Look at `num_nodes` and
  `num_edges` first.
- It was developed on simulations of around a thousand agents. Bigger networks should
  work, but the modularity step gets slow.
- α depends on where the power-law fit starts (`x_min`), and no alternative
  distribution (log-normal, say) is tested against it. Treat it as a descriptor.
- Holm's correction is applied per metric. If you test many metrics and want to control
  the error across all of them, adjust further yourself.
- The tests assume runs are independent. If your runs share a random seed or a starting
  state, the p-values will be too optimistic.
- The version is 0.x, so the API may still change.

## Reproducing the thesis analyses

The `scripts/` folder and the notebook are the thesis-specific layer: paths, condition
labels and the choice of figures for the two datasets used there (the Voat-like forum
runs from Tomašević et al., and the CNR baseline and recommender-system runs). The raw
simulation files are large and are not in the repository; `data/` only keeps the folder
layout.

```bash
git clone https://github.com/fabrifurna/sdt-netval
cd sdt-netval
pip install -e ".[dev]"

python scripts/05_stage_b_hypothesis.py cnr        # tests, CNR data
python scripts/06_cnr_comparison_visualization.py  # figure, CNR data
python scripts/05_stage_b_hypothesis.py legacy     # tests, Voat data
```

Scripts 01 to 04 do the same for the legacy dataset. Scripts 02 and 04 have their
condition labels written in, so they are not meant to be reused as they are.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

The tests build small synthetic SQLite, CSV and ZIP files on the fly, so no
simulation data is needed.

Layout of the package:

```text
src/sdt_netval/
├── adapters/    loader.py         reading .sqlite / .zip / .csv
├── core/        metrics.py        GraphMetrics
│                stats.py          mean and confidence interval
├── pipeline/    stage_a.py, stage_b.py
├── analysis/    hypothesis.py     Mann-Whitney, Holm, Cliff's delta
├── viz/         comparison.py     baseline vs conditions figure
├── export/      json_export.py
└── cli.py
```

## Citing

If this is useful in your work, `CITATION.cff` has the details GitHub needs to
generate a citation.

## References

- Giulio Rossetti, Massimo Stella, Rémy Cazabet, Katherine Abramski, Erica Cau,
  Salvatore Citraro, Andrea Failla, Riccardo Improta, Virginia Morini and Valentina
  Pansanella. *Y Social: an LLM-powered Social Media Digital Twin*. August 2024.
  arXiv:2408.00818.
- Aleksandar Tomašević, Darja Cvetković, Sara Major, Slobodan Maletić, Miroslav
  Anelković, Ana Vranić, Boris Stupovski, Dušan Vudragović, Aleksandar Bogojević and
  Marija Mitrović Dankulov. *Towards Operational Validation of LLM-Agent Social
  Simulations: A Replicated Study of a Reddit-like Technology Forum*. December 2025.
  arXiv:2508.21740.

## License

MIT, see `LICENSE`.

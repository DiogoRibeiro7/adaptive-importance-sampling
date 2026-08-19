# Notebooks

Four notebooks, executed with their outputs saved, so the plots and numbers
render directly on GitHub without running anything.

| Notebook | What it shows |
| --- | --- |
| [01_getting_started.ipynb](01_getting_started.ipynb) | The problem the estimator solves, one run on a benchmark, and a comparison against crude Monte Carlo |
| [02_benchmarks.ipynb](02_benchmarks.ipynb) | All five problems from the paper, each against an independently obtained reference |
| [03_high_dimensions.ipynb](03_high_dimensions.ipynb) | Accuracy from `d=2` to `d=200`, against the exact chi-square tail |
| [04_how_it_works.ipynb](04_how_it_works.ipynb) | The four moving parts — smoothed indicator, sigma schedule, penalised EM, heavy-tailed component |
| [05_flood_risk_real_data.ipynb](05_flood_risk_real_data.ipynb) | A real problem: 95 years of USGS gauge data, a levee, and three estimators that agree |
| [06_comparing_estimators.ipynb](06_comparing_estimators.ipynb) | All four estimators on the same problems, and where each one stops |

## A note on the numbers

Every figure quoted is measured when the notebook runs, not typed in. Where a
reference appears it comes from somewhere other than this package: a closed
form where one exists, otherwise crude Monte Carlo at a sample count the
estimator is meant to make unnecessary.

The estimator is stochastic, so the notebooks report a spread across seeds
rather than a single run. On the four-mode benchmark twelve independent runs
land between 0.83x and 1.19x of the reference; quoting the best one would
misrepresent it.

## Editing them

Each notebook is paired with a `.py` file in
[jupytext](https://jupytext.readthedocs.io/) percent format. The `.py` is the
one to edit — it reviews as a normal diff, where an `.ipynb` diff is mostly
base64 image data.

```bash
pip install jupytext
jupytext --to ipynb --execute 01_getting_started.py
```

That regenerates the `.ipynb` with fresh outputs. Run it from this directory,
with the package installed:

```bash
pip install -e ".[viz]"
```

## Runtime

All six take a few minutes in total. The slowest cells are the ones that have
to be slow to make their point: `02` runs the heat transfer problem, where each
limit-state evaluation is a finite-element solve, and `03` goes to 200
dimensions. `05` reads its data from `data/`, so it needs no network access.

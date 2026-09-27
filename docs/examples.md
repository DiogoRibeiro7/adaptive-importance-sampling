# Examples

## Notebooks

Six notebooks live in
[`notebooks/`](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/tree/main/notebooks).
They are committed with their outputs, so the plots and numbers render on
GitHub without running anything. Every number in them is measured when the
notebook runs, and every reference comes from outside the package.

| Notebook | What it shows |
| --- | --- |
| [01 Getting started](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/notebooks/01_getting_started.ipynb) | The problem the estimator solves, one run on a benchmark, and a comparison against crude Monte Carlo |
| [02 Benchmarks](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/notebooks/02_benchmarks.ipynb) | All five problems from the paper, each against an independently obtained reference |
| [03 High dimensions](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/notebooks/03_high_dimensions.ipynb) | Accuracy from \(d=2\) to \(d=200\), against the exact chi-square tail |
| [04 How it works](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/notebooks/04_how_it_works.ipynb) | The four moving parts: smoothed indicator, sigma schedule, penalized EM and the heavy-tailed component |
| [05 Flood risk, real data](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/notebooks/05_flood_risk_real_data.ipynb) | 95 years of USGS gauge data on the Potomac, a levee, and three estimators that agree |
| [06 Comparing estimators](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/notebooks/06_comparing_estimators.ipynb) | All four estimators on the same problems, and where each one stops working |

The flood notebook is the one to read if you have a problem of your own. It fits
distributions to a real record, maps them to standard normal space with
[`MarginalTransform`][safe_ice.MarginalTransform], and checks Safe-ICE against
crude Monte Carlo and subset simulation. It then shows why the answer is less
certain than the estimate: 95 years of data cannot tell a Gumbel tail from a
lognormal one, and the two disagree by a factor of nine.

To run them yourself, install the package with the `viz` extra and the
`benchmark` dependency group, which provides Jupyter and jupytext:

```bash
pip install -e ".[viz]"
pip install --group benchmark
jupyter lab notebooks/
```

Each notebook is paired with a jupytext `.py` file, which is the one to edit;
see [Development](development.md#notebooks).

## Scripts

The [`examples/`](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/tree/main/examples)
directory holds standalone scripts. Run them from the repository root.

| Script | What it does |
| --- | --- |
| `basic_usage.py` | Estimates a sphere problem and compares the result with the exact probability |
| `benchmark_comparison.py` | Runs Safe-ICE and crude Monte Carlo on the paper's benchmarks and plots the comparison |
| `high_dimensional.py` | Linear, nonlinear and sphere limit states from \(d=2\) to \(d=100\), plotting the estimate, sample count, run time and iterations against dimension |
| `advanced_features.py` | [`AdaptiveSafeICE`][safe_ice.AdaptiveSafeICE], time-variant, system, random-field and network problems, and the interactive plots when Plotly is installed |
| `performance_benchmark.py` | [`SafeICE`][safe_ice.SafeICE] against [`OptimizedSafeICE`][safe_ice.OptimizedSafeICE], timing and memory; needs the `perf` extra |

```bash
python examples/basic_usage.py
```

`quickstart.py`, also at the repository root, checks that the package imports
and runs a short estimate. It is a quick way to confirm an installation works.

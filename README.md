# Safe-ICE

[![CI](https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![arXiv](https://img.shields.io/badge/arXiv-2509.07160-b31b1b.svg)](https://arxiv.org/abs/2509.07160)

A Python implementation of **Safe Cross-Entropy-Based Importance Sampling** for
rare event simulation in reliability analysis.

Safe-ICE estimates failure probabilities `P_F = P(g(U) ≤ 0)` where `U ~ N(0, I)`
and `g` is a limit-state function. It targets problems where the failure
probability is far too small for crude Monte Carlo to resolve at a practical
sample count.

> **Status: alpha.** Two correctness bugs that biased every estimate have been
> fixed, and the estimator now tracks known analytical answers to within about
> 10%. See [Accuracy](#accuracy) for the numbers and what is still open.

Implements the method described in:

> Gao, Z., & Karniadakis, G. (2025). *Safe Cross-Entropy-Based Importance
> Sampling for Rare Event Simulations.* arXiv:2509.07160.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Reproducibility](#reproducibility)
- [Further usage](#further-usage)
- [Accuracy](#accuracy)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Installation

Requires Python 3.11 or newer.

```bash
pip install git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
```

The package is not on PyPI yet. For a local checkout:

```bash
git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
cd adaptive-importance-sampling-ice
pip install -e .
```

### Optional extras

| Extra  | Adds                           | For                                                     |
| ------ | ------------------------------ | ------------------------------------------------------- |
| `viz`  | plotly, seaborn, pandas        | `safe_ice.analysis.interactive_visualization`           |
| `perf` | numba, psutil, memory-profiler | `OptimizedSafeICE` JIT paths, memory instrumentation    |
| `all`  | everything above               |                                                         |

```bash
pip install -e ".[viz]"
```

## Quick start

```python
from safe_ice import BenchmarkProblems, SafeICE

# A limit-state function: failure occurs where g(u) <= 0.
g = BenchmarkProblems.four_mode_series_system(z=2.0)

ice = SafeICE(
    limit_state_function=g,
    dimension=2,
    K0=8,  # initial mixture components
    N=1000,  # samples per iteration
    delta_star=1.5,  # CV stopping threshold
    random_state=0,  # omit for non-deterministic runs
)

pf, results = ice.run(verbose=True)

print(f"Failure probability: {pf:.6e}")
print(f"Iterations run:      {len(results['iterations'])}")
print(f"Final components:    {results['final_components']}")
```

### Using your own limit-state function

`g` receives an `(n, d)` array of points and should return the matching `(n,)`
array of values. Failure is `g(u) <= 0`.

```python
import numpy as np

from safe_ice import SafeICE


def my_limit_state(u: np.ndarray) -> np.ndarray:
    """Fails when the point leaves a radius-3 ball."""
    return 3.0 - np.linalg.norm(u, axis=-1)


pf, results = SafeICE(my_limit_state, dimension=10, random_state=0).run()
```

Scalar-only functions also work: if a batch call fails, Safe-ICE falls back to
evaluating one row at a time. That fallback is far slower, so prefer a
vectorised function.

### What `run()` returns

`run()` returns `(pf, results)`, where `results` contains:

| Key                   | Type             | Meaning                                           |
| --------------------- | ---------------- | ------------------------------------------------- |
| `iterations`          | `list[dict]`     | One record per iteration (`K`, `sigma`, `lambda`) |
| `final_components`    | `int`            | Mixture components remaining at convergence       |
| `final_samples`       | `ndarray (n, d)` | Samples from the final proposal                   |
| `final_weights`       | `ndarray (n,)`   | Importance weights for those samples              |
| `final_g_values`      | `ndarray (n,)`   | Limit-state values for those samples              |
| `convergence_metrics` | `dict`           | `cv_values` and `delta_values` per iteration      |

`iterations` is a list of records, so the iteration *count* is
`len(results["iterations"])`.

## How it works

Safe-ICE combines three ideas:

1. **A vMF-Nakagami mixture proposal.** Points are decomposed into a radius and
   a direction. The radius follows a Nakagami distribution, the direction a
   von Mises-Fisher distribution.
2. **Penalized EM.** A cross-entropy penalty on the mixture weights drives
   redundant components toward zero, so the component count adapts instead of
   being fixed up front.
3. **A heavy-tailed safety component.** The proposal is
   `q_safe = λ·q_vMFNM + (1-λ)·q_heavy`, where `q_heavy` uses an inverse
   Nakagami radius. It keeps mass in the far tail so the search cannot collapse
   onto a single mode. A cosine annealing schedule moves `λ` from exploration
   toward exploitation.

Each iteration samples from `q_safe`, evaluates `g`, compares the coefficient
of variation of the importance weights against `delta_star`, adapts the
smoothing parameter `σ`, and refits the mixture by penalized EM.

## Reproducibility

Every estimator accepts `random_state`, which takes an `int`, a
`numpy.random.Generator`, or `None`:

```python
import numpy as np

from safe_ice import SafeICE

SafeICE(g, dimension=2, random_state=42)  # seeded
SafeICE(g, dimension=2, random_state=np.random.default_rng(0))  # your generator
SafeICE(g, dimension=2)  # global NumPy state
```

`None` uses NumPy's global random state, so results then depend on whatever
else has drawn random numbers in the same process. Seed your runs when
comparing results.

`PerformanceEvaluator.run_monte_carlo_reference` accepts `random_state` too.

## Further usage

### Comparing against crude Monte Carlo

```python
from safe_ice import BenchmarkProblems, PerformanceEvaluator

g = BenchmarkProblems.four_mode_series_system()
evaluator = PerformanceEvaluator()

pf_mc, std_mc = evaluator.run_monte_carlo_reference(
    limit_state_func=g, dimension=2, n_samples=1_000_000, random_state=0
)

comparison = evaluator.compare_methods(
    limit_state_func=g,
    dimension=2,
    reference_pf=pf_mc,
    n_runs=10,
    safe_ice_params={"K0": 6, "N": 1000},
)
```

### Convergence analysis

```python
from safe_ice import AdvancedAnalysis

analyzer = AdvancedAnalysis()
analyzer.analyze_component_evolution(results)
analyzer.analyze_sample_distribution(results, g)  # 2D problems only
```

### Command line

```bash
safe-ice --version
safe-ice demo
safe-ice benchmark --problem four-mode
```

### Available benchmark problems

`four_mode_series_system`, `three_mode_problem`,
`two_mode_opposite_directions`, `nonlinear_oscillator`,
`nonlinear_oscillator_simplified`, and `nakagami_ratio_problem`, plus
`HeatTransferProblem` — a PDE problem using a Karhunen-Loève expansion.

## Accuracy

Two correctness bugs that biased every estimate have been fixed:

- The heavy-tailed component of the proposal was missing the polar Jacobian
  `r^(d-1)`, so it was not a probability density: it integrated to `2.7` at
  `d=2` and `114` at `d=5`. Since it sits in the importance-sampling
  denominator, every estimate was scaled by that error, which is why results
  came out at roughly `0.54x` the analytical answer no matter how large `N`
  was.
- The cosine annealing schedule drove `λ` to exactly `1.0`, removing the
  heavy-tailed component from the proposal entirely. That component is the
  "safe" part of Safe-ICE: it keeps the proposal's tails heavier than the
  target so the weights stay bounded. Without it the estimate could be
  dominated by a single sample — on one seed, 99.5% of the estimate came from
  one point. `λ` is now capped by `lambda_max` (default `0.95`).

Current accuracy against problems with known answers, median over seeds:

| Problem | Reference | Estimate / reference |
| --- | --- | --- |
| Linear limit state, closed form | `1.69e-2` | `1.08` |
| Sphere `d=2`, closed form | `1.11e-2` | `1.02` |
| Four-mode series system, 2e7-sample MC | `5.8e-5` | `1.10` |

`tests/test_proposal_normalisation.py` pins this down: it integrates each
proposal component numerically and fails if the Jacobian is dropped again.

### Remaining limitations

- **Estimates are not constrained to `[0, 1]`.** The estimator is unbiased but
  unconstrained, so for a limit state that fails everywhere (true probability
  exactly 1) it scatters around 1 and can land slightly above. Clamping would
  fix it, but that is a modelling decision rather than a bug.
- **High-dimensional variance is large.** At `d=10` with a small iteration
  budget, estimates still vary considerably across seeds.
- The reference value quoted for the four-mode problem used to be `1.22e-5`.
  Crude Monte Carlo over 2e7 samples puts it at `5.8e-5 ± 1.7e-6` for the
  default `z=3.8`.

## Development

```bash
git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
cd adaptive-importance-sampling-ice

pip install -e .
pip install --group dev          # pip >= 25.1; or: uv sync --group dev
pre-commit install
```

| Task       | Command                                   |
| ---------- | ----------------------------------------- |
| Fast tests | `pytest`                                  |
| All tests  | `pytest -m ""`                            |
| Lint       | `ruff check .`                            |
| Format     | `ruff format .`                           |
| Type check | `mypy`                                    |
| Coverage   | `pytest --cov=safe_ice --cov-report=html` |

`pytest` skips tests marked `slow` by default so the common case stays quick;
CI runs the full set. Ruff replaces black, isort, and flake8 — there is no
separate formatter to install.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Citation

If you use this software, please cite both the implementation (see
[CITATION.cff](CITATION.cff)) and the original paper:

```bibtex
@article{gao2025safe,
  title   = {Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations},
  author  = {Gao, Zhiwei and Karniadakis, George},
  journal = {arXiv preprint arXiv:2509.07160},
  year    = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- The original method is by Zhiwei Gao and George Karniadakis.
- Builds on the ICE method of Papaioannou et al. (2019) and the cross-entropy
  method of Rubinstein & Kroese (2004).

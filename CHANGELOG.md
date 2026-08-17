# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `random_state` on `PerformanceEvaluator.run_monte_carlo_reference`, so
  reference runs are reproducible.
- `safe_ice/typing.py` with shared `LimitStateFunction`, `NDArrayF`, `RNGLike`
  and `SeedLike` aliases.
- `tests/conftest.py` providing seeded `rng` and `seed` fixtures; every test
  that draws random numbers now uses them instead of NumPy's global state.
- `tests/test_numeric_helpers.py` pinning the vectorised density code to the
  scalar reference formulas.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, issue templates and a
  pull request template.
- A "Known limitations" section in the README recording measured accuracy gaps.

### Removed

- The `architecture-diagrams` workflow, its `docs/architecture/` output, and
  `benchmarks/architecture_config.py`. The workflow was 1826 lines, 62% of all
  the workflow code in the repository, and regenerated diagrams on every push
  that touched a Python file, committing them straight to `main` with
  `[skip ci]`. Nothing linked to the output. Of the four diagram types, only the
  class diagram produced anything real: `module-diagram.mmd` was an empty
  `graph LR`, `architecture-overview.mmd` was a single generic
  "Business Logic Layer" node, and the generated README linked an
  `architecture-overview.svg` that was never produced. It also carried handling
  for JavaScript, TypeScript and Java. `architecture_config.py` was template
  boilerplate for a web application, listing `User`, `Order` and `Product` as
  the important classes and grouping modules by `routes`/`controllers`.

### Changed

- Dependency updates are handled by Dependabot alone. The custom
  `auto-upgrade-pyproject` workflow duplicated it, had been silently broken
  since packaging moved to PEP 621, and would have produced PRs competing with
  Dependabot's over the same constraints. Dependabot now groups routine minor
  and patch bumps into one PR and uses `increase-if-necessary`, so declared
  floors only move when a release actually requires it.
- Added benchmarks under `benchmarks/` covering the vectorised densities and a
  short end-to-end run, so the performance regression workflow has something to
  measure. A normal `pytest` run is unaffected.

- **Packaging consolidated onto `pyproject.toml`.** Metadata moved to PEP 621
  `[project]`; `setup.py`, `setup.cfg`, `mypy.ini` and `MANIFEST.in` are gone.
  They disagreed with each other on version, classifiers, and dependency
  floors, and `MANIFEST.in` was ignored by the build backend anyway.
- **Supported Python is now 3.11 - 3.14** (was 3.9 - 3.12). Dependency floors
  moved to `numpy>=1.26`, `scipy>=1.11`, `matplotlib>=3.8`. The previous
  `numpy ^2.4.1` pin could not be satisfied on the Python versions the package
  claimed to support.
- Benchmark and profiling packages (`pytest-benchmark`, `memory-profiler`,
  `psutil`) moved out of the runtime dependencies into dev/optional groups.
- Development dependencies are declared as PEP 735 `[dependency-groups]`.
- `__version__` is now read from installed package metadata, so the version is
  stated once in `pyproject.toml` rather than in three places.
- Ruff replaces black, isort and flake8 for both linting and formatting.
- CI now runs ruff, `ruff format --check`, mypy, the full test suite on four
  Python versions, a packaging check that installs the built wheel, and the
  security scanners. It previously ran only a syntax-level ruff subset.
- The release workflow builds and publishes a GitHub Release, and verifies the
  tag matches the packaged version. PyPI publishing is intentionally not wired
  up yet.
- Version bumping moved from an automatic push-to-`main` job to a manual
  workflow that opens a pull request.
- `Dockerfile` no longer installs Poetry to export requirements; it installs the
  package with pip into a virtualenv.
- Slow tests are marked `slow` and excluded from the default `pytest` run.

### Fixed

- **The heavy-tailed proposal component was missing the polar Jacobian.** It is
  built as a radial density times an angular one, so recovering a density on
  R^d needs `du = r^(d-1) dr dw`, which `vMFNMDistribution.pdf` applied and
  this did not. The component integrated to 2.7 at d=2 and 114 at d=5, and
  since it sits in the importance-sampling denominator every estimate was
  scaled by that error. On a linear limit state with a closed-form answer the
  estimate was 0.54x the analytical value regardless of sample size; it is now
  1.02x. `OptimizedSafeICE` had the same omission in both of its components.
- **The von Mises-Fisher sampler was wrong for concentrations above 30.** A
  shortcut intercepted `kappa >= 30` and replaced Wood's rejection sampler with
  a Gaussian around `mu` of scale `0.2 / sqrt(kappa)`. The 0.2 is arbitrary and
  roughly five times too small (the tangent-space standard deviation is
  `1 / sqrt(kappa)`), and it ignores the dimension, so samples clustered far too
  tightly around `mu`. Measured against the closed-form mean resultant length
  `I_{d/2}(k)/I_{d/2-1}(k)`, at d=20 and kappa=50 it gave 0.9925 against a true
  0.8263. Wood's sampler is exact at any concentration, accurate to ~1e-6 out to
  kappa = 1e4 at constant cost, so the shortcut only introduced error. Removing
  it also improved the estimator, since for `kappa >= 30` the sampler and the
  density it is weighted by no longer agreed: the sphere problem went from 0.86
  to 1.01 of the closed form at d=10, and 0.54 to 0.76 at d=20.
- **The Nakagami sampler drew from the wrong distribution for shapes above
  100.** A normal approximation replaced the exact `sqrt(Gamma(m, Omega/m))`
  route there. Its variance, `Omega*(1 - 1/(4m))/m`, is about `Omega/m`, while
  the Nakagami variance tends to `Omega/(4m)`, so samples came out with exactly
  twice the correct spread. A KS test against `scipy.stats.nakagami` rejected at
  p = 0 for every shape above the cutoff, and passes at p > 0.1 now. The second
  moment stayed within 0.3% of `Omega`, so a moment check alone did not reveal
  it. `InverseNakagamiDistribution.sample` inherited the error, since it draws
  R and returns 1/R.
- **The Nakagami density was badly wrong for shape parameters above 170.** A
  normal approximation replaced the exact form there, on the theory that the
  exact one would overflow. It does not: log-space with `loggamma` is stable for
  any shape. Measured against `scipy.stats.nakagami`, the approximation was off
  by 43% at m=200 and 145% at m=500, where the log-space form is accurate to
  ~1e-13. The CDF had the same branch, off by ~5e-3 absolute, plus a cap on the
  incomplete-gamma argument that was never needed. This is reachable rather
  than hypothetical, because the initialiser now sets the shape from the problem
  dimension.
- Removed `safe_ice/distributions/nakagami_stable.py`. It duplicated the live
  module under the same class names and was referenced by nothing, but it was
  also the *correct* implementation for large shapes, so its behaviour was
  folded into `nakagami.py` before deleting it. The only thing lost is a
  `log_pdf` helper that nothing called.
- **The mixture was initialised with dimension-independent parameters.** The
  target is the standard normal in R^d, whose radius follows chi_d, and chi_d
  is exactly Nakagami(m = d/2, Omega = d), so the fixed values used before were
  only appropriate near d=2. At d=20 the initial proposal sat at radius ~1.1
  while the target sits at ~4.4; the two barely overlapped and the run never
  recovered, returning ~1e-20 for a true 1.5e-2. Initialisation now scales with
  the dimension, and d=20 goes from 0 of 6 seeds usable to 6 of 6.
- **The cross-entropy penalty in the EM step had its sign inverted.**
  Equation 21 uses `[ln pi_k - sum_s pi_s ln pi_s]`, and `sum_s pi_s ln pi_s`
  is minus the entropy; the code subtracted the entropy instead. At uniform
  weights the bracket should be 0 but evaluated to -2 ln K, subtracting a flat
  0.3 from every weight at K=20 and zeroing all but the largest component on
  the first EM step. The mixture collapsed to one component every run, which
  defeats the automatic component selection the penalty exists to provide. It
  now settles on 2-4 components for the four-mode problem, and the estimate
  improves from 1.08x to 1.01x of the reference.
- **Cosine annealing drove lambda to exactly 1.0**, which removed the
  heavy-tailed component from the proposal altogether. That component is what
  keeps the proposal's tails heavier than the target so importance weights stay
  bounded; without it a single tail sample could carry the estimate (99.5% of
  it on one seed, giving 0.71 for a 5.8e-5 event). Lambda is now capped by a
  new `lambda_max` parameter, default 0.95, matching the cap `AdaptiveSafeICE`
  already applied. On the four-mode benchmark, 4 of 12 seeds landed within 2x
  of the true value before; all 12 do now.
- The four-mode benchmark test cited a reference of 1.22e-5. Crude Monte Carlo
  over 2e7 samples puts it at 5.8e-5 +/- 1.7e-6 for the default z=3.8.

- **`PerformanceEvaluator.compare_methods` crashed** with `TypeError` because it
  treated `results["iterations"]` (a list of per-iteration records) as a count.
- **The README's quick-start example printed a list of dicts** where it claimed
  to print an iteration count.
- A duplicated `'y'` key in the plotly slider configuration silently discarded
  the first value.
- `scripts/pyproject_editor.py` detected the wrong layout for a pyproject that
  has both `[project]` and `[tool.poetry]`, which is the standard Poetry 2.x
  shape; it also crashed printing non-ASCII on legacy Windows code pages.
- `scripts/pyproject_updater.py` contained a `return latest.major ==
  latest.major` tautology and an if/else whose branches were identical.
- `Dockerfile`, `Dockerfile.docs` and `Dockerfile.jupyter` all referenced files
  or commands that no longer exist (`setup.py`, `poetry export`, a `notebooks/`
  directory, `jupyter labextension install`).
- Tests no longer depend on NumPy's global random state, so results do not
  change with test ordering. One test was passing only because an unrelated
  test had called `np.random.seed(42)` earlier in the session.

### Performance

- **Wood's vMF sampler is vectorised.** It drew one point per Python-loop
  iteration, with a separate Householder rotation each time, costing about
  200 microseconds per sample. Rejection now happens in batches and the
  rotation is applied to every row at once. Drawing 20000 samples at d=20,
  kappa=50 goes from 1.74 s to 0.065 s; at d=10 from 4.20 s to 0.047 s. The
  default test suite drops from 26 s to 12 s and the full suite from 118 s to
  89 s. Verified statistically identical to the per-sample version by a
  two-sample KS test (p = 0.29 to 0.86), and the batched rotation matches the
  scalar one to 4e-16. The now-unreferenced scalar helpers are removed rather
  than left to drift out of step.

- **Roughly 200x faster.** `vMFNMDistribution.pdf`, the penalized-EM E-step and
  the heavy-tailed density each looped in Python over every (sample, component)
  pair, calling scalar PDFs; `NakagamiDistribution.pdf` alone was called 4.2
  million times in a single two-iteration run. They now evaluate one vectorised
  pass per component. A 2-D problem at `N=1000, max_iterations=5` went from
  108.7 s to 0.5 s, and the full test suite from over 20 minutes (never
  completing) to about 70 s.
- Results are unchanged to within ~1e-13 relative, verified against captured
  baselines from the previous implementation across 48 parameter combinations
  and separately for the heavy-tailed density across 45 more.
- The inverse-Nakagami `Omega` in the heavy-tailed density was recomputed once
  per sample although it only depends on the component; it is now hoisted.

### Known limitations

Recorded as `xfail` tests, not silenced:

- The estimator does not reliably reproduce the reference probability on the
  four-mode series benchmark (6 of 12 seeds land in range; estimates span
  `6e-6` to `5e-1` against a reference of `1.22e-5`).
- On a linear limit state with a closed-form answer, the estimate is
  systematically about `0.54x` the analytical value, and the gap does not
  shrink with more samples.
- Estimates are not constrained to `[0, 1]`.

## [0.1.0] - 2025-01-27

### Added

- First public release.
- Core Safe-ICE algorithm with penalized EM component selection.
- vMF-Nakagami mixture distribution and inverse Nakagami heavy tails.
- Benchmark problems from the paper and a heat transfer problem using a
  Karhunen-Loève expansion.
- Performance evaluation and convergence analysis utilities.

[Unreleased]: https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/releases/tag/v0.1.0

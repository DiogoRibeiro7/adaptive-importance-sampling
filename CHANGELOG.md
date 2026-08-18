# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The search for sigma in equation (10) was ill-posed, and runs collapsed
  because of it.** The equation minimises `(CV(W_t(sigma)) - delta_target)^2`
  over `(0, sigma_prev)`, and the whole interval was handed to a bounded
  minimiser. Below some sigma the smoothed indicator `Phi(-g/sigma)` underflows
  to zero for every sample, so the weights all vanish and the CV is undefined;
  on a rare-event problem that is most of the interval, and a minimiser given a
  mostly-infinite objective returns arbitrary points. On the heat transfer
  problem it took sigma from 1 to 0.373, where the CV is 18.75, when 0.999 was
  available at 9.5 against a target of 4. Sigma then fell faster than the
  proposal could follow, the CV never recovered, and two of six seeds ended at
  1e-27 and 1e-34 against a reference of 4.69e-07.

  Simply fixing the search to find the true global minimum is worse. Just above
  the underflow region only a handful of samples still carry weight, and the CV
  of one surviving sample out of N is about `sqrt(N)`, so it sweeps through the
  target on its way to infinity. Those spurious minima sit at a sigma hundreds
  of times too small, and are what the old minimiser was accidentally missing.

  The CV rises monotonically as sigma falls, so the minimiser is the largest
  sigma at which it reaches the target. That is now bracketed from above and
  found by bisection. The densities in `W_t` do not depend on sigma, so they
  are hoisted out of the search, which costs about as much as the single
  objective evaluation it replaces.

  Accuracy is unchanged and robustness is much better. Across nine problems the
  median estimate is within 0.93x to 1.07x of its reference, and the *worst
  individual seed* is 0.83x:

  | Problem | reference | median | worst seed |
  | --- | --- | --- | --- |
  | four-mode `z=1.0` | `6.465e-05` | 1.00x | 0.83x |
  | three-mode `z=3.0` | `3.475e-03` | 1.05x | 0.96x |
  | two-mode `z=3.0` | `2.700e-03` | 1.07x | 0.95x |
  | oscillator `z=0.05` | `1.798e-03` | 0.98x | 0.88x |
  | nakagami ratio | `5.188e-02` | 1.00x | 0.98x |
  | sphere `d=2` | `2.187e-03` | 1.02x | 1.00x |
  | sphere `d=10` | `5.346e-03` | 1.01x | 1.01x |
  | sphere `d=50` | `3.610e-03` | 0.99x | 0.95x |
  | sphere `d=200` | `4.565e-03` | 0.93x | 0.85x |

  The heat transfer estimate is now `3.07e-07` against the paper's `4.69e-07`
  on every seed, where before the median hid two collapsed runs.

- **The heat transfer problem did not work, and no test would have noticed.**
  It was the least covered module in the package and had never been checked
  against an independent answer. The temperature field came from an explicit
  relaxation, `T += 0.01 * (laplacian + Q)`, run for 1000 sweeps and clamped to
  +/-1e6. An elliptic problem has no time derivative to march, and that fixed
  pseudo-step was about sixteen times the explicit stability limit for the
  default grid, so it diverged: 380 of 441 nodes ended pinned at the clamp and
  the limit state returned exactly `threshold` for every input, with no
  dependence on the random field at all.

  Four parameters also disagreed with section 4.5: the correlation length was
  `0.5` against `l = 0.2`, the threshold `100` against equation (48)'s `10`,
  the covariance `exp(-r/l)` against equation (46)'s `exp(-r^2/l^2)`, and the
  heat source's y extent was computed and then discarded, leaving a
  full-height strip instead of the square `A`.

  The field is now a direct sparse solve of the five-point conservative
  discretisation, which reproduces the analytic solution of the uniform case to
  machine precision. Safe-ICE estimates the failure probability at `2.83e-07`
  against the paper's `4.69e-07`, a factor of 0.60, using finite differences on
  21x21 where the paper uses finite elements over 25040 triangles.

  Two further defects came out of the same work. Region membership used exact
  floating-point comparisons on bounds that a `linspace` reproduces at some
  grid sizes and misses by an ulp at others, so a region silently lost a row of
  nodes at `grid_size=31`. And assigning `Q` to each source node made the
  discrete heat input `((0.1 + h) / 0.1)^2` times too large -- 2.25x at
  `grid_size=21` against 1.44x at 51 -- which was the whole of the problem's
  apparent grid dependence. Spreading `Q` over the region's area fixes it: the
  nominal average temperature on `B` is now 4.553 at every grid size tested,
  against 10.24, 8.09, 7.11 and 6.56 before.

  The conductivity exponent is clipped. The estimator's heavy-tailed proposal
  reaches far enough into the field's tail to underflow `kappa` to zero, which
  leaves the conduction matrix singular and the solve returning NaN, failing
  the run outright. The bounds are absurd for a field with mean 1 and standard
  deviation 0.3, so realistic samples are untouched.

  The Karhunen-Loeve modes are now sign-canonicalised. An eigenvector is only
  defined up to sign and LAPACK's choice varies between builds, so the same
  coefficients produced mirror-image conductivity fields on different machines;
  a test asserting a direction passed locally and failed in CI. Anchoring each
  mode on its largest-magnitude entry makes the expansion reproducible.

  Boundary conditions follow Figure 11 -- zero Dirichlet on the top edge, zero
  Neumann on the other three. The text of section 4.5 says the opposite, and
  the two cannot both hold: with three edges able to shed heat the nominal
  average temperature on `B` is 0.49 against a threshold of 10, which is 51
  standard deviations away and could never produce the reported probability.

- **The penalty coefficient did not follow equation (23).** `_update_beta`
  called itself a "simple adaptive heuristic" and was one: it measured each
  weight's deviation from the mean rather than its change since the previous
  iteration, scaled that by the number of components rather than the number of
  samples, replaced equation (24)'s entropy term with `(1 - max pi) / min pi`,
  and blended the result with the previous beta. Beta governs how aggressively
  redundant components are pruned, which is one of the paper's two headline
  contributions. It now implements equations (23) and (24).
- **Component pruning did not follow equation (22).** Components were dropped
  at `pi > 1e-4` after clamping negatives to zero and renormalising. Equation
  (21) is a zero-sum redistribution, so it already preserves the normalisation
  and can drive a weight negative; the paper prunes exactly those. It now does.
- **The stopping rule of section 3.2 was missing.** The criterion reads the
  vMFNM samples, so when `lambda` is 0 there are none and the CV is set to
  infinity to force another iteration. This was approximated by an ad-hoc
  "at least two iterations" guard, which is now unnecessary: `lambda_0` is
  always 0 because `M = sigma0`.
- `em_max_iter` now defaults to 20, the value stated in section 4, rather
  than 100.

Accuracy is unchanged to slightly better, and pruning now does what it is for:
the mixture settles at K=1 for the unimodal sphere at d=50 and for the
oscillator, K=8 for the four-mode problem, from K0=20.

| Problem | reference | median | ratio |
| --- | --- | --- | --- |
| four-mode `z=1.0` | `6.465e-05` | `6.446e-05` | 1.00x |
| three-mode `z=3.0` | `3.475e-03` | `3.641e-03` | 1.05x |
| two-mode `z=3.0` | `2.700e-03` | `2.879e-03` | 1.07x |
| oscillator `z=0.05` | `1.798e-03` | `1.773e-03` | 0.99x |
| sphere `d=10` | `5.346e-03` | `5.410e-03` | 1.01x |
| sphere `d=50` | `3.610e-03` | `3.565e-03` | 0.99x |

### Documented

- The `lambda_max` cap is recorded as a deliberate divergence from equation
  (35), which reaches exactly 1. Section 3.2 relies on that to argue the final
  estimate is sound, but it removes the heavy-tailed component that makes the
  method "safe" and leaves the weights unbounded.
- `nakagami_ratio_problem` is not one of the paper's benchmarks. Sections 4.1
  to 4.5 cover the four-mode, three-mode, oscillator, two-mode and heat
  transfer problems; this is an extra exercise for the Nakagami code.

- **The four-mode benchmark had the wrong constant, and every reference
  measured against it was wrong too.** The last two branches of equation (37)
  are `u1 - u2 + 7/sqrt(2)`; the code had `sqrt(3.5)`, reading the fraction
  `7/sqrt(2)` as `sqrt(7/2)` -- 1.8708 instead of 4.9497. That placed the
  failure region far closer to the origin: crude Monte Carlo gives `1.88e-01`
  at `z=0`, where the paper reports about `1e-03`, which is above the top of
  its Figure 4 axis. The corrected function gives `2.22e-03` there.

  This invalidates the reference correction recorded below under 0.2.0, which
  replaced a quoted `1.22e-5` with a measured `5.815e-05`: that measurement was
  taken against the broken limit state. Corrected values from 2e7 samples:

  | `z` | `pf` | rel. s.e. |
  | --- | --- | --- |
  | 0.0 | `2.2188e-03` | 0.5% |
  | 0.5 | `4.1245e-04` | 1.1% |
  | 1.0 | `6.4650e-05` | 2.8% |
  | 1.5 | `9.3500e-06` | 7.3% |
  | 2.0 | `1.0500e-06` | 21.8% |

  The default `z` moves from 3.8 to 1.0. 3.8 was chosen against the broken
  function; 1.0 is one of the thresholds tabulated in the paper's Table 1, and
  is rare enough to be a meaningful benchmark while still having a Monte Carlo
  reference. Safe-ICE estimates it at `6.543e-05`, 1.01x the reference.

- **The nonlinear oscillator benchmark now implements the paper's model.** It
  computed the displacement as `force_rms / (k * (1 - alpha))`, a closed form
  appearing nowhere in the source and which never integrated the equations of
  motion. It produced values around `4e-07` against a threshold of `0.05`, so
  the problem could not fail: crude Monte Carlo over 2e7 samples found zero
  failures and every estimator returned exactly `0`. It was recorded as a
  strict `xfail` and withdrawn from the CLI.

  It now integrates the Bouc-Wen hysteretic oscillator of Section 4.3
  (equations 39-42) with fourth-order Runge-Kutta at `dt = 0.01`, and
  reproduces Figure 7 of the paper:

  | `z` | this implementation | paper |
  | --- | --- | --- |
  | 0.05 | `1.798e-03` | ~`1.5e-03` |
  | 0.06 | `1.475e-04` | ~`1.2e-04` |
  | 0.07 | `4.5e-06` | ~`5e-06` |

  Safe-ICE estimates the first at `1.797e-03`, within 1% of the Monte Carlo
  value, from 1000 samples per iteration against 2e6 for crude Monte Carlo.
  The benchmark is available from the CLI again.

  The limit-state function now rejects input of the wrong dimension instead of
  zero-padding it, which had been hiding a test that passed 2-column input to
  a 10-dimensional problem.

### Removed

- `BenchmarkProblems.nonlinear_oscillator_simplified`. It existed only as the
  closed-form approximation described above; there is no simplified variant in
  the paper. Use `nonlinear_oscillator`.

### Changed

- **`run()` now returns a value constrained to `[0, 1]`.** The estimator is
  unbiased but unconstrained -- the weights `p(u)/q_safe(u)` are a ratio of
  densities and nothing bounds them by 1 -- so a single run could return a
  value above 1. Callers expect a probability, so the returned value is
  clamped.

  Clamping truncates overshoots without touching undershoots, which biases the
  mean downwards: on a limit state that fails everywhere, the mean over 40 runs
  goes from 1.0163 raw to 0.8811 clamped. It never fires in the rare-event
  regime this package is for, where estimates sit two to four orders of
  magnitude below 1.

  An estimate above 1 is therefore treated as a diagnostic rather than a
  nuisance: it means the proposal is not covering the target and the weights
  have gone degenerate, so a few samples carry the whole sum. Returning `1.0`
  silently would present such a run as converged. The raw value is kept in
  `results["pf_unclamped"]` and a `RuntimeWarning` explains the cause.

  `test_certain_failure` was a non-strict `xfail` because of this; it is now a
  real test.

## [0.2.0] - 2026-08-18

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

- The `numba` dependency and the `perf` extra's JIT claim. The two `@jit`
  kernels in `safe_ice_optimized.py` were defined but never called, so the
  extra pulled in a large dependency that did nothing. `perf` now provides
  memory instrumentation only.
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

- **`OptimizedSafeICE` and `AdaptiveSafeICE` returned `nan` for every seed on
  every benchmark problem.** Both are exported from the package root. They
  reimplemented the algorithm rather than reusing it, and the copy did not
  work: adaptation weights used a hard indicator, which is zero for every
  sample until one lands in the failure region, so on a rare-event problem
  every weight was zero, the elite set was empty and the loop stopped after one
  iteration; the smoothing parameter was moved by ±10% according to whether the
  number of mixture components had changed, rather than by targeting the weight
  coefficient of variation, so it rose from 1.00 to 1.21 over a run instead of
  falling towards the failure region; and the final estimate divided by the sum
  of those already-zeroed weights, which was exactly zero. `AdaptiveSafeICE`
  additionally initialised the radial parameters from fixed ranges regardless
  of dimension, placing the proposal at radius ~1.5 whether the problem was 2-
  or 200-dimensional.

  Both now subclass `SafeICE` and override only sample generation, so the
  sigma schedule, the weights and the estimator are shared with the
  implementation that is tested against Monte Carlo. On the four-mode problem
  the three agree at 0.94x, 0.98x and 1.17x of the reference; on the sphere
  problem `OptimizedSafeICE` matches the exact chi-square tail to within 3% at
  d = 10, 30 and 50. Vectorised sampling is retained and is now a real
  optimisation: 3.5x faster than `SafeICE` at N = 4000.

- **The nonlinear oscillator benchmark cannot fail, and so measures nothing.**
  Displacement is computed as `force_rms / (k * (1 - alpha))` with `k = 5e6`,
  giving values around 4e-7 against a threshold of `z = 0.05`. Reaching it needs
  a norm of about 7.3e5, where a 10-dimensional standard normal averages 3.1.
  Crude Monte Carlo over 2e7 samples finds zero failures, and `safe-ice
  benchmark oscillator` reported a probability of exactly 0 with an infinite
  coefficient of variation. The scaling is inconsistent by roughly 1e5.
  Reconstructing the intended formulation needs the source paper, so the defect
  is documented on the problem, recorded as a strict xfail, and the problem is
  removed from the CLI in favour of `two-mode`.
- Reference probabilities in the CLI were wrong: four-mode was quoted as 1.22e-5
  against a measured 5.8e-5, and three-mode as 2.3e-3 against 3.5e-3. Both now
  come from the Monte Carlo values recorded in the tests. The CLI epilogue also
  pointed at a `your-username` placeholder URL.

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
- **The proposal density was floored at 1e-15 before being divided into the
  prior.** That is not a small number for a probability density on R^d, which
  shrinks geometrically with dimension. Measured on the sphere problem, the
  median proposal density is 4.5e-2 at d=2, 1.6e-8 at d=10, 4.3e-15 at d=20 and
  7.1e-22 at d=30, so the floor clamped nothing at low dimension, 11% of samples
  at d=20, and every sample at d=30. Clamping inflates the denominator and
  collapses the estimate: d=30 returned about 1e-9 against a true 1.6e-2.

  This presented as a hard ceiling between d=20 and d=30 and was twice
  attributed here to importance-weight degeneracy. That is a real phenomenon,
  but it was the wrong diagnosis: with the floor at the smallest positive float,
  the sphere problem holds to within a few percent through d=200. The same fixed
  floor was also applied inside two log-likelihood computations, capping every
  term at log(1e-15) and flattening the EM objective in high dimensions.

  Sphere problem, median over seeds, ratio to the exact chi-square tail:

    d=20   0.75 -> 1.01
    d=30   0.00 -> 0.98   (was 7e-9 against 1.6e-2)
    d=50   0.00 -> 0.98
    d=200    --  -> 1.01

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

- Estimates are not constrained to `[0, 1]`. The estimator is unbiased but
  unconstrained, so on a limit state that fails everywhere it scatters around
  the true value of 1 and can land above it. Clamping would fix it at the cost
  of introducing bias, which is a modelling decision. See `ROADMAP.md`.
- The nonlinear oscillator benchmark cannot fail as parameterised, so it
  measures nothing. Fixing the scaling requires the source paper.

The two accuracy gaps previously listed here -- the four-mode benchmark
reproducing its reference on only 6 of 12 seeds, and a systematic `0.54x` bias
on a linear limit state -- were both symptoms of the bugs fixed in this
release, and no longer occur. The four-mode reference itself was wrong: it was
quoted as `1.22e-5` against a measured `5.815e-05`.

## 0.1.0 - 2025-01-27

### Added

- First public release.
- Core Safe-ICE algorithm with penalized EM component selection.
- vMF-Nakagami mixture distribution and inverse Nakagami heavy tails.
- Benchmark problems from the paper and a heat transfer problem using a
  Karhunen-Loève expansion.
- Performance evaluation and convergence analysis utilities.

[Unreleased]: https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/releases/tag/v0.2.0

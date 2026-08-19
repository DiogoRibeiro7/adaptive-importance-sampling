# Roadmap

Where the project is and what is planned next. Items are listed in the order
they are likely to be worked on, not by importance.

## Now: 0.3.0

0.3.0 is a second correctness release, and the one that checked the
implementation against the source paper equation by equation. It fixed the
four-mode limit state, which had `sqrt(3.5)` where equation (37) has
`7/sqrt(2)`; implemented the nonlinear oscillator and the heat transfer
problem, neither of which worked; replaced the penalty coefficient with
equations (23) and (24); and repaired the search for sigma in equation (10),
which had been minimising an objective that is undefined over most of its
domain. Coverage went from 53% to 85%.

0.2.0 was the first release since the package was reorganised, also a
correctness release: eight bugs that biased or broke every estimate were found
by comparing against independent references, and the packaging, CI and test
suite were rebuilt around catching that class of problem earlier.

The estimator now agrees with closed-form answers to within a few percent from
`d=2` through `d=200`, and with 2e7-sample Monte Carlo on the benchmark
problems. See [Accuracy](README.md#accuracy) for the measured numbers and
[CHANGELOG.md](CHANGELOG.md) for the full list.

### Decided: estimates are clamped to `[0, 1]`

Returned values are clamped, so `run()` always yields a probability. The raw
estimate stays in `results["pf_unclamped"]` and an overshoot raises a
`RuntimeWarning`, because a value above 1 means the proposal is not covering
the target and the run should be treated as unconverged rather than as a
probability of 1. Clamping never fires in the rare-event regime, where
estimates sit two to four orders of magnitude below 1.

### Done: the nonlinear oscillator benchmark works

It previously computed the displacement in closed form and never integrated the
equations of motion, so it could not fail and every estimator returned 0. It
now implements the paper's Bouc-Wen model (equations 39-42) with fourth-order
Runge-Kutta, and reproduces Figure 7: 1.798e-03 at `z=0.05`, 1.475e-04 at
`0.06` and 4.5e-06 at `0.07`. Safe-ICE estimates the first to within 1% of the
Monte Carlo value. It is available from the CLI again.

### Done: the heat transfer problem is solved and referenced

Its field solver diverged and the limit state ignored its input entirely. It
now uses a direct sparse solve, verified against an analytic solution, and
Safe-ICE estimates `2.83e-07` against the paper's `4.69e-07`.

### Done: the sigma schedule no longer collapses

Equation (10)'s search was ill-posed. Below some sigma the smoothed indicator
underflows for every sample and the CV is undefined, which on a rare-event
problem is most of the interval; a bounded minimiser handed a mostly-infinite
objective returned arbitrary points. It is now bracketed from above, and the
worst individual seed across nine problems sits at 0.83x its reference, where
two of six seeds on the heat transfer problem previously returned 1e-27.

### Done: every module is now covered

Line coverage is 85%, from 53% when the roadmap was written. The three modules
that were at 0% were each resolved on their merits rather than by writing
tests for whatever was there:

* the advanced problem generators turned out to be correct, so they were
  exported and pinned to closed-form answers (89%);
* `utils/performance.py` was removed. Nothing imported it, and where it
  overlapped `distributions/_numeric.py` it was the weaker of the two --
  `norm * exp(kappa * dot)` overflows at `kappa = 800` where the log-space
  form in use does not;
* the Plotly visualiser has smoke tests (73%). It draws rather than computes,
  so a defect there misleads rather than corrupts, and the tests are shallow
  by design.

## Next

### More estimators, for comparison and cross-checking

`ICEvMFNM` is in, which is the baseline the paper's tables compare against.
Two more were agreed:

`SubsetSimulation` is in as well, and has already paid for itself: it
independently confirmed that the heat transfer gap against the paper is the
finite-difference discretisation and not the estimator. One method remains:

* **Cross-entropy with a Gaussian mixture** (Kurtz and Song 2013, reference
  [25]), the older variant ICE improved on. A third point of comparison.

### Real data, which needs an isoprobabilistic transform first

The estimator works in standard normal space, and real inputs are not standard
normal, so there is no way to apply it to measured data at present. The paper
notes the same thing in its introduction: a Nataf or Rosenblatt transformation
maps the original distributions to Gaussian ones. That layer has to exist
before any real-data example can.

Agreed example: river flood exceedance from USGS daily discharge records --
fit marginals to a real gauge record, transform, and estimate the probability
of exceeding a flood stage.


These are the next items from a repository review on 2026-08-18. The estimator
core is in much better shape than the surrounding product surface; the highest
return is now to make the public API, docs, visualisation tools and release
metadata line up with the corrected implementation.

### Fix the result object contract

`run()` currently returns internally useful data under names that promise
something narrower than they contain. `results["final_samples"]` and
`results["final_g_values"]` are all iteration samples, not the separate sample
set used for the final probability estimate, and `results["final_weights"]` is
an array of ones rather than the final importance weights. That is harmless for
tests that only check shape and non-negativity, but it makes downstream plots
and user analyses compute the wrong thing while looking plausible.

Define the result schema explicitly before adding more analysis features:

* keep per-iteration proposal samples separate from final-estimator samples;
* expose the actual final importance weights used in equation (36);
* record `n_failures`, `parameters`, `sigma`, `lambda`, `cv` and the current
  estimate consistently for each iteration, or remove consumers that expect
  them;
* update README, Sphinx docs and tests to assert semantics, not just key
  presence.

### Repair the analysis and visualisation layer

The numerical core is now covered; the plotting layer mostly has smoke tests.
Several functions still depend on stale result fields:

* `AdvancedAnalysis.analyze_component_evolution` reads `history["lambda"]`,
  but `SafeICE` records `history["lambda_val"]`;
* the Plotly convergence view hard-codes a target CV of `0.05`, while the
  algorithm default is `delta_star = 1.5`;
* the failure-count plot reads `n_failures`, which `run()` never records;
* the mixture-evolution animation expects per-iteration `parameters`, which are
  not stored;
* the dashboard computes a displayed probability from placeholder weights
  rather than from the importance-sampling estimator.

Fix these after the result schema is settled. Tests should check plotted values
against a small deterministic run, not only that a figure object exists.

### Refresh the public documentation

The docs build cleanly, but some examples still describe the pre-0.3 API and
old benchmark state. In particular, `docs/source/quickstart.rst` still cites
the old four-mode reference (`1.22e-5`), reads `iter_data["delta"]` although
iteration records expose `sigma`, and imports `VisualizationTools`, which is
not part of the package. The install docs also still recommend Poetry commands
even though project metadata and development groups now live in PEP 621/735.

Bring README, Sphinx docs and examples back into one story: corrected
benchmarks, current result keys, current dependency installation, and the
actual analysis API.

### Normalise repository identity before release

GitHub reports that the repository has moved from
`adaptive-importance-sampling-ice` to `adaptive-importance-sampling`. The old
URL still appears in package metadata, docs, badges, Docker labels, citation
metadata, Zenodo metadata and the CLI epilogue. Either keep the old URL
deliberately as a redirect, or update all public links in one release-prep
commit so badges, installation snippets, citation metadata and changelog links
point at the canonical repository.

### Finish 0.3.0 release hygiene

The release branch already bumps `pyproject.toml`, `CITATION.cff`,
`.zenodo.json` and the conda recipe to `0.3.0`, and the version-bump tooling
now knows about `.zenodo.json`. Before tagging:

* move the populated `CHANGELOG.md` `0.3.0` section out of any local-only state;
* verify `python scripts/pyproject_editor.py --check bump-version patch`
  reports companion-file changes without writing them;
* run the full CI-equivalent suite with an installed package:
  `ruff check .`, `ruff format --check .`, `mypy`,
  `pytest -m "" -n auto`, Sphinx with `-W`, build, and `twine check`;
* decide whether `.zenodo.json` should carry `version` even though Zenodo's
  legacy JSON schema does not list it. Zenodo's GitHub integration documents
  version metadata, but editor schema validation may complain if a strict
  legacy schema is attached.

### Preserve benchmark evidence outside the tests

The package now relies on independent references: closed forms, 2e7-sample
Monte Carlo for 2D benchmarks, 2e6-sample Monte Carlo for the oscillator, and a
paper comparison for heat transfer. Those numbers are too expensive to
regenerate in normal CI, so the project should keep the scripts, seeds,
sample counts and resulting confidence intervals as auditable artefacts under
`benchmarks/` or `paper/`. That makes future correctness releases easier to
review and keeps README/CHANGELOG claims reproducible.

### Improve local validation ergonomics

`ruff` and formatting are fast, mypy passes, and docs build with warnings as
errors. The default local test run is still long enough that it is easy to hit
tooling timeouts, while CI gets parallelism from `pytest -n auto`. Make the
local workflow clearer by documenting the intended quick, full and release
commands, and consider adding a Makefile target that mirrors CI exactly after
installing the dev group.

## Later

### Publish to PyPI

Deliberately not wired up. The release workflow builds a GitHub Release and
verifies the tag matches the packaged version. When the project is ready,
register it as a PyPI Trusted Publisher and add a `pypa/gh-action-pypi-publish`
job with `id-token: write`; no API token is needed.

### Restore Docker image publishing

The `build-docker` job was dropped because it referenced `DOCKER_USERNAME` and
`DOCKER_PASSWORD`, neither of which is set on this repository, so it could only
fail. No image was ever published. Bringing it back needs the two secrets and a
job using `docker/login-action` and `docker/build-push-action`.

### Implement `safe-ice analyze`

The subcommand currently prints a pointer to the Python API.

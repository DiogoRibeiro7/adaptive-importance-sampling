# Roadmap

Where the project is and what is planned next. Items are listed in the order
they are likely to be worked on, not by importance.

## Now: 0.2.0

The first release since the package was reorganised. It is a correctness
release: eight bugs that biased or broke every estimate were found by comparing
against independent references, and the packaging, CI and test suite were
rebuilt around catching that class of problem earlier.

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

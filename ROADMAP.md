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

## Next

### Decide whether estimates should be clamped to `[0, 1]`

The estimator is unbiased but unconstrained, so on a limit state that fails
everywhere (true probability exactly 1) it scatters around 1 and can land above
it. Clamping would make every return value a valid probability, at the cost of
making the estimator biased. This is a modelling decision rather than a bug, so
it is recorded as an `xfail` on `test_certain_failure` rather than silently
resolved either way.

### Fix or remove the nonlinear oscillator benchmark

`nonlinear_oscillator` and `nonlinear_oscillator_simplified` cannot fail as
parameterised: displacement comes out around `4e-7` against a threshold of
`0.05`, so reaching failure needs a norm of about `7.3e5` where a
10-dimensional standard normal averages `3.1`. Crude Monte Carlo over 2e7
samples finds zero failures. The scaling is inconsistent by roughly `1e5`.
Reconstructing the intended formulation needs the source paper, so the defect
is recorded (`test_should_be_a_usable_rare_event_benchmark`, a strict `xfail`)
rather than guessed at. It has been withdrawn from the CLI in the meantime.

### Cover the untested modules

Overall line coverage is 53%, but it is not evenly spread. The core algorithm,
the distributions and the optimiser are at 87-94%. Four modules are not
meaningfully exercised at all:

| Module | Coverage | Notes |
| --- | --- | --- |
| `problems/advanced_problems.py` | 0% | 238 statements, four problem classes, not exported from the package |
| `analysis/interactive_visualization.py` | 0% | Requires the `viz` extra |
| `utils/performance.py` | 0% | |
| `problems/heat_transfer.py` | 11% | Exported publicly; has no reference-based check |

Every module examined closely so far has turned out to contain a defect that
changed results, so these are treated as unverified rather than as working.
`advanced_problems.py` needs a decision first: it is unreachable through the
public API, so it is either exported and tested or removed.

### Give `HeatTransferProblem` a reference

It is exported from the package root but has no check against an independent
answer, which is the gap that hid the benchmark defects until they were
measured against Monte Carlo.

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

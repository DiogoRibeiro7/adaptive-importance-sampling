# Accuracy and benchmarks

An estimator of small probabilities is only useful if its estimates can be
trusted, and a plausible-looking wrong number is worse than none. Safe-ICE is
checked against answers obtained without it: closed forms where they exist,
otherwise crude Monte Carlo at a sample count the estimator is meant to make
unnecessary, or an independent method.

## Accuracy against known answers

Median over seeds of the estimate divided by the reference:

| Problem | Reference | Estimate / reference |
| --- | ---: | ---: |
| Linear limit state, closed form | \(1.69\times10^{-2}\) | 1.02 |
| Sphere \(d=2\), closed form | \(1.11\times10^{-2}\) | 1.01 |
| Sphere \(d=10\), closed form | \(5.35\times10^{-3}\) | 1.00 |
| Sphere \(d=20\), closed form | \(1.54\times10^{-2}\) | 1.01 |
| Sphere \(d=50\), closed form | \(3.61\times10^{-3}\) | 0.98 |
| Sphere \(d=200\), closed form | \(4.57\times10^{-3}\) | 1.01 |
| Four-mode series system, crude Monte Carlo over \(2\times10^7\) | \(6.5\times10^{-5}\) | 1.01 |

The sphere problem fails when \(\lVert\mathbf{u}\rVert\) exceeds a radius, so its
exact probability is a chi-square tail, and it can be posed in any dimension.
[Notebook 03](../examples.md#notebooks) runs it from \(d=2\) to \(d=200\).

These are medians. A single run is one draw of a random estimate; on the
four-mode problem, seeds 0 to 5 at the default settings land between 0.79 and
1.09 of the reference.

## Benchmark problems

[`BenchmarkProblems`][safe_ice.BenchmarkProblems] provides the paper's test
problems with their default thresholds:

| Problem | \(d\) | Reference \(P_F\) | Source |
| --- | ---: | ---: | --- |
| [`four_mode_series_system`][safe_ice.BenchmarkProblems.four_mode_series_system] | 2 | \(6.465\times10^{-5}\) | Crude Monte Carlo, \(2\times10^7\) samples |
| [`three_mode_problem`][safe_ice.BenchmarkProblems.three_mode_problem] | 2 | \(3.475\times10^{-3}\) | Crude Monte Carlo, \(2\times10^7\) samples |
| [`two_mode_opposite_directions`][safe_ice.BenchmarkProblems.two_mode_opposite_directions] | any | \(2.700\times10^{-3}\) | Closed form, \(2\Phi(-3)\) |
| [`nonlinear_oscillator`][safe_ice.BenchmarkProblems.nonlinear_oscillator] | 10 | \(1.798\times10^{-3}\) | Crude Monte Carlo, \(2\times10^6\) samples |
| [`nakagami_ratio_problem`][safe_ice.BenchmarkProblems.nakagami_ratio_problem] | 2 | \(5.174\times10^{-2}\) | Closed form, \(\Phi(\ln 0.1/\sqrt{2})\) |
| [`HeatTransferProblem`][safe_ice.HeatTransferProblem] | 10 | \(4.69\times10^{-7}\) | The paper, by subset simulation over 50 runs |

Notes on individual problems:

- **Four-mode series system.** Failure is \(g(\mathbf{u}) + z \le 0\), so a
  larger `z` makes it rarer. Over the range of the paper's Figure 4, `z` from
  0 to 2, crude Monte Carlo puts \(P_F\) between \(2.2\times10^{-3}\) and
  \(1.05\times10^{-6}\).
- **Nonlinear oscillator.** A hysteretic Bouc-Wen oscillator under white-noise
  ground acceleration. Its reference comes from fewer samples than the others
  because every evaluation integrates the equations of motion. The paper varies
  the threshold `z` from 0.05 to 0.08, which takes \(P_F\) from
  \(1.8\times10^{-3}\) down to \(1.5\times10^{-7}\).
- **Nakagami ratio problem.** Not one of the paper's benchmarks; an extra
  exercise for the Nakagami machinery.
- **Heat transfer.** A steady heat equation with a lognormal conductivity field
  from a ten-term Karhunen-Loève expansion. Safe-ICE estimates
  \(2.83\times10^{-7}\), 0.60 of the paper's value, but on a 21×21
  finite-difference grid where the paper uses finite elements with 25,040
  triangles, so the two do not solve quite the same discretised problem.

## Correctness fixes behind these numbers

Earlier versions returned estimates that were consistently wrong, and the
tests now pin each cause down. The most consequential:

- **A proposal that was not a density.** The heavy-tailed component was missing
  the polar Jacobian \(r^{d-1}\), so it integrated to 2.7 at \(d=2\) and 114 at
  \(d=5\). It sits in the importance-sampling denominator, so every estimate was
  scaled by that error. `tests/test_proposal_normalisation.py` now integrates
  each component numerically.
- **A density floor that grew with dimension.** Proposal densities were floored
  at \(10^{-15}\) before division. Densities on \(\mathbb{R}^d\) shrink
  geometrically with \(d\), so this clamped 11% of samples at \(d=20\) and every
  sample at \(d=30\), which produced an apparent dimensional ceiling. The floor
  is now the smallest positive float.
- **Initialisation independent of dimension.** The radius of a standard normal
  in \(\mathbb{R}^d\) is \(\chi_d\), which is exactly
  \(\text{Nakagami}(m = d/2, \Omega = d)\). The mixture now starts there instead
  of at fixed values that were right only near \(d=2\).
- **A penalty with the wrong sign.** The cross-entropy penalty subtracted the
  entropy where equation 21 adds it, and zeroed all but one component on the
  first EM step. On the four-mode problem the mixture now settles on two to
  four components, as it should.
- **A safety component annealed away.** The cosine schedule drove \(\lambda\) to
  exactly 1, removing the heavy-tailed component. On one seed a single sample
  then carried 99.5% of the estimate. \(\lambda\) is now capped by
  `lambda_max`.
- **A benchmark with the wrong constant.** Two branches of the four-mode limit
  state read \(7/\sqrt{2}\) as \(\sqrt{7/2}\), which put the failure region far
  closer to the origin. Any reference measured against the old function measured
  a different problem.

The [changelog](../changelog.md) records each fix and how it was found.

## Known limitations

- **Estimates are clamped to \([0, 1]\).** Clamping truncates overshoots without
  touching undershoots, which biases the mean down by about 12% on a limit state
  that fails everywhere. It never fires on a rare-event problem, and when it
  fires it warns, as
  [Running and diagnosing](running.md#estimates-outside-0-1) explains.
- **One run is one draw.** Report a spread over seeds, not a single estimate.
- **The limit state needs a usable gradient.** See
  [Where Safe-ICE struggles](estimators.md#where-safe-ice-struggles).

The [roadmap](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/ROADMAP.md)
lists what is planned.

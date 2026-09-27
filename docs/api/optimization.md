# Penalized EM

[`SafeICE`][safe_ice.SafeICE] refits its mixture after every iteration with an
importance-weighted EM step. The cross-entropy penalty on the mixture weights
drives redundant components to zero, which is what lets the component count
adapt instead of being fixed up front. [Theory](../theory.md#penalized-em)
gives the update.

You only need this class directly to experiment with the fitting step itself;
the estimators construct their own.

::: safe_ice.PenalizedEMOptimizer

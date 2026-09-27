# Transforms

The estimators sample a standard normal prior, so a limit state written in
physical units has to be mapped into that space first.
[`MarginalTransform`][safe_ice.MarginalTransform] does this for independent or
correlated inputs, and [Limit-state functions](../guide/limit-states.md#problems-in-physical-units)
walks through an example.

::: safe_ice.MarginalTransform

::: safe_ice.transforms.Marginal

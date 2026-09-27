# Distributions

The proposal is built from three distributions: a von Mises-Fisher distribution
for the direction, and a Nakagami (light-tailed) or inverse Nakagami
(heavy-tailed) distribution for the radius. [Theory](../theory.md#the-proposal)
shows how they combine.

The samplers and densities are static methods: parameters are passed on every
call rather than to a constructor. Only
[`vMFNMDistribution`][safe_ice.vMFNMDistribution] is instantiated, because it
holds a [`vMFNMParameters`][safe_ice.vMFNMParameters].

!!! note "Densities include the polar Jacobian"

    [`vMFNMDistribution.pdf`][safe_ice.vMFNMDistribution.pdf] includes the polar
    Jacobian \(r^{d-1}\). Every component of the proposal must integrate to one,
    since it sits in the denominator of the importance weights;
    `tests/test_proposal_normalisation.py` checks this numerically.

::: safe_ice.VonMisesFisherSampler

::: safe_ice.NakagamiDistribution

::: safe_ice.InverseNakagamiDistribution

::: safe_ice.vMFNMDistribution

Distributions Module
====================

The distributions module implements the probability distributions used in
Safe-ICE. Three modules make up the public surface:

``safe_ice.distributions.vmf``
    :class:`~safe_ice.distributions.vmf.VonMisesFisherSampler`, the angular part
    of the proposal.

``safe_ice.distributions.nakagami``
    :class:`~safe_ice.distributions.nakagami.NakagamiDistribution` and
    :class:`~safe_ice.distributions.nakagami.InverseNakagamiDistribution`, the
    light- and heavy-tailed radial parts.

``safe_ice.distributions.mixture``
    :class:`~safe_ice.distributions.mixture.vMFNMDistribution`, the combined
    mixture used as the light-tailed proposal.

.. note::

   The samplers and densities are **static methods**: parameters are passed on
   every call rather than to a constructor. Only ``vMFNMDistribution`` is
   instantiated, because it holds a :class:`~safe_ice.core.parameters.vMFNMParameters`.

von Mises-Fisher
----------------

.. automodule:: safe_ice.distributions.vmf
   :members:
   :undoc-members:
   :show-inheritance:

Nakagami and Inverse Nakagami
-----------------------------

.. automodule:: safe_ice.distributions.nakagami
   :members:
   :undoc-members:
   :show-inheritance:

vMF-Nakagami Mixture
--------------------

.. automodule:: safe_ice.distributions.mixture
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

von Mises-Fisher Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Draws directions on the unit sphere, concentrated around ``mu``.

.. code-block:: python

   import numpy as np

   from safe_ice.distributions.vmf import VonMisesFisherSampler

   mu = np.array([0.0, 0.0, 1.0])  # north pole
   kappa = 10.0                    # high concentration

   samples = VonMisesFisherSampler.sample(mu, kappa, 1000)

   assert samples.shape == (1000, 3)
   assert np.allclose(np.linalg.norm(samples, axis=1), 1.0)

Nakagami Distribution
~~~~~~~~~~~~~~~~~~~~~

The radial part of the light-tailed proposal. Note that ``chi_d`` is exactly
``Nakagami(m=d/2, Omega=d)``, which is how the mixture is initialised.

.. code-block:: python

   import numpy as np

   from safe_ice.distributions.nakagami import NakagamiDistribution

   m, omega = 2.0, 1.0

   r = np.linspace(0.01, 3.0, 100)
   pdf_values = NakagamiDistribution.pdf(r, m, omega)
   cdf_values = NakagamiDistribution.cdf(r, m, omega)

   samples = NakagamiDistribution.sample(m, omega, 1000)

   # E[R^2] = Omega
   assert abs(np.mean(samples**2) - omega) < 0.2

Heavy-Tailed Sampling
~~~~~~~~~~~~~~~~~~~~~

The inverse Nakagami is the radial part of the safety component: it keeps mass
in the far tail so the importance weights stay bounded.

.. code-block:: python

   import numpy as np

   from safe_ice.distributions.nakagami import InverseNakagamiDistribution

   m, omega = 2.0, 1.0

   samples = InverseNakagamiDistribution.sample(m, omega, 1000)

   y = np.linspace(0.1, 10.0, 100)
   pdf_values = InverseNakagamiDistribution.pdf(y, m, omega)

   # Tails decay far more slowly than the Nakagami's
   print(f"density at y=10: {pdf_values[-1]:.2e}")

vMFNM Mixture
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from safe_ice.core.parameters import vMFNMParameters
   from safe_ice.distributions.mixture import vMFNMDistribution

   mu = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
   params = vMFNMParameters(
       pi=np.array([0.6, 0.4]),       # mixture weights, must sum to 1
       m=np.array([2.0, 3.0]),        # Nakagami shape
       Omega=np.array([1.0, 1.5]),    # Nakagami scale
       mu=mu,                         # component directions, unit rows
       kappa=np.array([5.0, 8.0]),    # concentrations
   )

   dist = vMFNMDistribution(params)

   samples = dist.sample(1000)
   densities = dist.pdf(samples)
   total_log_likelihood = dist.log_likelihood(samples)

   assert samples.shape == (1000, 3)
   assert np.all(densities > 0)

.. note::

   ``pdf`` returns a density on ``R^d``, including the polar Jacobian
   ``r^(d-1)``. Every component of the proposal must integrate to one, since it
   sits in the denominator of the importance weights;
   ``tests/test_proposal_normalisation.py`` checks this numerically.

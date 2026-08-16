Core Module
===========

The core module contains the main Safe-ICE algorithm implementation.

SafeICE Class
-------------

.. automodule:: safe_ice.core.safe_ice
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: vMFNMParameters

Parameters
----------

.. automodule:: safe_ice.core.parameters
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic usage of the SafeICE class:

.. code-block:: python

   from safe_ice import SafeICE
   import numpy as np

   # Define limit state function
   def g(u):
       return 3.0 - np.linalg.norm(u, axis=-1)

   # Create SafeICE instance
   ice = SafeICE(
       limit_state_function=g,
       dimension=2,
       N=1000,
       max_iterations=10
   )

   # Run algorithm
   pf, results = ice.run(verbose=True)

   # Access results
   print(f"Failure probability: {pf:.2e}")
   print(f"Number of iterations: {len(results['iterations'])}")
   print(f"Final number of components: {results['iterations'][-1]['K']}")

Advanced Configuration
----------------------

Configure algorithm parameters:

.. code-block:: python

   ice = SafeICE(
       limit_state_function=g,
       dimension=10,
       N=2000,                # Samples per iteration
       max_iterations=20,     # Maximum iterations
       K0=30,                 # Initial components
       delta_target=3.0,      # Target rarity
       delta_star=1.5,        # Rarity increment
       sigma0=1.5,            # Initial sigma
       em_max_iter=100        # EM iterations
   )

Working with Parameters
-----------------------

Create and use custom initial parameters:

.. code-block:: python

   from safe_ice.core.parameters import vMFNMParameters
   import numpy as np

   # Create parameters
   params = vMFNMParameters(
       K=3,
       d=2,
       pi=np.array([0.4, 0.3, 0.3]),
       m=np.array([2.0, 2.5, 3.0]),
       Omega=np.array([1.0, 1.2, 0.8]),
       mu=np.random.randn(3, 2),
       kappa=np.array([5.0, 3.0, 4.0])
   )

   # Normalize directions
   params.mu /= np.linalg.norm(params.mu, axis=1, keepdims=True)

   # Use with SafeICE
   pf, results = ice.run(initial_params=params)
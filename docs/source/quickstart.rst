Quickstart Guide
================

This guide will help you get started with Safe-ICE quickly.

Installation
------------

Install Safe-ICE using pip:

.. code-block:: bash

   pip install git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git

Or install from source:

.. code-block:: bash

   git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
   cd adaptive-importance-sampling-ice
   pip install -e .

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

Here's a simple example using a spherical limit state function:

.. code-block:: python

   import numpy as np
   from safe_ice import SafeICE

   # Define a limit state function
   def sphere_limit_state(u):
       """Failure when ||u|| > 3"""
       return 3.0 - np.linalg.norm(u, axis=-1)

   # Initialize Safe-ICE
   ice = SafeICE(
       limit_state_function=sphere_limit_state,
       dimension=2,
       N=1000,               # Samples per iteration
       max_iterations=10     # Maximum iterations
   )

   # Run the algorithm
   pf, results = ice.run(verbose=True)

   # Print results
   print(f"Failure probability: {pf:.2e}")
   print(f"Total samples used: {len(results['final_samples'])}")

Using Benchmark Problems
~~~~~~~~~~~~~~~~~~~~~~~~~

Safe-ICE includes several benchmark problems from the literature:

.. code-block:: python

   from safe_ice.problems.benchmarks import BenchmarkProblems

   # Load benchmark problems
   problems = BenchmarkProblems()

   # Four-mode series system (2D, pf ≈ 1.22e-5)
   g = problems.four_mode_series_system()

   # Run Safe-ICE
   ice = SafeICE(g, dimension=2, N=1000)
   pf, results = ice.run()

Available benchmarks:

* ``four_mode_series_system()``: Classic 2D reliability problem
* ``three_mode_problem()``: Three-mode failure problem
* ``nonlinear_oscillator()``: 10D nonlinear oscillator
* ``nakagami_ratio_problem()``: Ratio of Nakagami variables
* ``two_mode_opposite_directions(dimension)``: Configurable dimension

Custom Limit State Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can define your own limit state functions:

.. code-block:: python

   def custom_limit_state(u):
       """
       Custom nonlinear limit state function.

       Parameters
       ----------
       u : array_like
           Input samples, shape (n_samples, dimension) or (dimension,)

       Returns
       -------
       g_values : array_like
           Limit state values, g(u) < 0 indicates failure
       """
       # Ensure 2D array
       if u.ndim == 1:
           u = u.reshape(1, -1)

       # Example: nonlinear combination
       term1 = 3.5 - 0.1 * (u[:, 0] - u[:, 1])**2
       term2 = (u[:, 0] + u[:, 1]) / np.sqrt(2)

       return term1 - term2

   # Use custom function
   ice = SafeICE(custom_limit_state, dimension=2, N=500)
   pf, results = ice.run()

Advanced Configuration
----------------------

Adjusting Algorithm Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ice = SafeICE(
       limit_state_function=g,
       dimension=10,
       N=2000,                    # More samples per iteration
       max_iterations=20,         # More iterations
       K0=30,                     # Initial mixture components
       delta_target=2.0,          # Target rarity parameter
       delta_star=1.0,            # Intermediate rarity
       sigma0=1.5,                # Initial sigma
       em_max_iter=100            # EM iterations
   )

Using Initial Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

You can provide initial parameters for the vMFNM distribution:

.. code-block:: python

   from safe_ice.core.parameters import vMFNMParameters
   import numpy as np

   # Create initial parameters
   initial_params = vMFNMParameters(
       K=3,                              # 3 components
       d=2,                              # 2D problem
       pi=np.array([0.5, 0.3, 0.2]),   # Mixture weights
       m=np.array([2.0, 2.5, 3.0]),     # Nakagami m parameters
       Omega=np.array([1.0, 1.2, 0.8]), # Nakagami Omega parameters
       mu=np.random.randn(3, 2),        # vMF mean directions
       kappa=np.array([5.0, 3.0, 4.0])  # vMF concentrations
   )

   # Normalize mu to unit vectors
   initial_params.mu /= np.linalg.norm(initial_params.mu, axis=1, keepdims=True)

   # Run with initial parameters
   pf, results = ice.run(initial_params=initial_params)

Analyzing Results
-----------------

The ``results`` dictionary contains detailed information:

.. code-block:: python

   # Access results
   samples = results['final_samples']      # All generated samples
   weights = results['final_weights']      # Importance weights
   g_values = results['final_g_values']    # Limit state values

   # Convergence metrics
   metrics = results['convergence_metrics']
   cv_values = metrics['cv_values']        # Coefficient of variation
   delta_values = metrics['delta_values']  # Rarity parameters

   # Iteration details
   iterations = results['iterations']      # Per-iteration data
   for i, iter_data in enumerate(iterations):
       print(f"Iteration {i}: K={iter_data['K']}, delta={iter_data['delta']:.2f}")

Visualization
~~~~~~~~~~~~~

Visualize results for 2D problems:

.. code-block:: python

   from safe_ice.analysis.visualization import VisualizationTools

   viz = VisualizationTools()

   # Plot convergence
   viz.plot_convergence(results)

   # Analyze sample distribution (2D only)
   viz.analyze_sample_distribution(results, g)

Command-Line Interface
----------------------

Safe-ICE includes a CLI for quick demonstrations:

.. code-block:: bash

   # Run demo
   safe-ice demo

   # Run benchmark
   safe-ice benchmark four-mode --samples 2000 --iterations 15

   # List available benchmarks
   safe-ice benchmark --list

Next Steps
----------

* Read the :doc:`theory` section to understand the algorithm
* Explore :doc:`examples/index` for detailed use cases
* Check the :doc:`api/core` for complete API reference
* See :doc:`api/distributions` for the distributions used by the proposal
Examples
========

This section provides detailed examples of using Safe-ICE for various applications.

Runnable versions of these live in the ``examples/`` directory of the
repository.

Quick Examples
--------------

Simple 2D Problem
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from safe_ice import SafeICE

   # Simple sphere limit state
   def g(u):
       return 3.0 - np.linalg.norm(u, axis=-1)

   # Run Safe-ICE
   ice = SafeICE(g, dimension=2, N=1000)
   pf, results = ice.run()

   print(f"Failure probability: {pf:.2e}")

Benchmark Problem
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from safe_ice.problems.benchmarks import BenchmarkProblems

   # Load benchmark
   problems = BenchmarkProblems()
   g = problems.four_mode_series_system()

   # Run with more samples
   ice = SafeICE(g, dimension=2, N=2000, max_iterations=15)
   pf, results = ice.run(verbose=True)

High-Dimensional Problem
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 10D nonlinear oscillator
   g = problems.nonlinear_oscillator()

   ice = SafeICE(
       g,
       dimension=10,
       N=3000,
       max_iterations=20,
       K0=30  # More initial components
   )

   pf, results = ice.run()

Custom Problem with Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from safe_ice.analysis.visualization import AdvancedAnalysis

   # Define custom problem
   def custom_g(u):
       # Nonlinear limit state
       term1 = 4.0 - 0.1 * (u[:, 0]**2 + u[:, 1]**2)
       term2 = 0.5 * (u[:, 0] - u[:, 1])
       return term1 - term2

   # Run Safe-ICE
   ice = SafeICE(custom_g, dimension=2, N=1500)
   pf, results = ice.run()

   # Visualize results
   analyzer = AdvancedAnalysis()
   analyzer.analyze_component_evolution(results)
   analyzer.analyze_sample_distribution(results, custom_g)
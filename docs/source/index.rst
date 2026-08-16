.. Safe-ICE documentation master file

Safe-ICE Documentation
======================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

**Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations**

Safe-ICE is a cutting-edge Python implementation of the Safe Cross-Entropy (Safe-CE) method for rare event simulation. It features automatic component selection through penalized optimization and efficient sampling using von Mises-Fisher Nakagami mixtures (vMFNM).

.. note::
   This documentation is for Safe-ICE version |version|.

Key Features
------------

* **Automatic Component Selection**: Penalized EM algorithm automatically determines optimal mixture components
* **Heavy-Tailed Distributions**: Enhanced exploration using inverse-Nakagami tail behavior
* **Efficient Sampling**: von Mises-Fisher mixtures on unit sphere with radial Nakagami distributions
* **Benchmark Problems**: Includes standard reliability analysis test problems
* **Type Safety**: Comprehensive type hints throughout the codebase
* **Well-Tested**: Extensive test suite with >80% coverage

Quick Example
-------------

.. code-block:: python

   from safe_ice import SafeICE
   from safe_ice.problems.benchmarks import BenchmarkProblems

   # Load a benchmark problem
   problems = BenchmarkProblems()
   g = problems.four_mode_series_system()

   # Initialize and run Safe-ICE
   ice = SafeICE(
       limit_state_function=g,
       dimension=2,
       N=1000,
       max_iterations=10
   )

   pf, results = ice.run(verbose=True)
   print(f"Failure probability: {pf:.2e}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   theory

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/distributions

Development
-----------

Contribution guidelines and the changelog live in the repository:
`CONTRIBUTING.md
<https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/blob/main/CONTRIBUTING.md>`_
and `CHANGELOG.md
<https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/blob/main/CHANGELOG.md>`_.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use Safe-ICE in your research, please cite:

.. code-block:: bibtex

   @article{safece2024,
     title={Safe Cross-Entropy Method for Rare Event Simulation},
     author={...},
     journal={...},
     year={2024}
   }
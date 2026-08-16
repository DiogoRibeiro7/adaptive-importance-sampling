Installation Guide
==================

This guide covers different ways to install Safe-ICE.

Requirements
------------

* Python 3.11 or higher (tested up to 3.14)
* NumPy >= 1.26
* SciPy >= 1.11

Installation Methods
--------------------

Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Safe-ICE is not published on PyPI yet, so install it from the repository.

.. code-block:: bash

   pip install git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git

Install with optional dependencies:

.. code-block:: bash

   # Include visualization tools
   pip install "safe-ice[viz] @ git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git"

   # Include all optional dependencies
   pip install "safe-ice[all] @ git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git"

From Source
~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
   cd adaptive-importance-sampling-ice

   # Install in development mode
   pip install -e .

   # Or with poetry
   poetry install

Using Poetry
~~~~~~~~~~~~

If you use Poetry for dependency management:

.. code-block:: bash

   # Add to your project
   poetry add safe-ice

   # Or clone and install
   git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
   cd adaptive-importance-sampling-ice
   poetry install

Using Conda
~~~~~~~~~~~

Create a conda environment and install:

.. code-block:: bash

   # Create environment
   conda create -n safe-ice python=3.11
   conda activate safe-ice

   # Install dependencies
   conda install numpy scipy matplotlib

   # Install Safe-ICE
   pip install git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git

Docker Installation
~~~~~~~~~~~~~~~~~~~

Use the provided Docker image:

.. code-block:: bash

   # No image is published yet, so build it locally
   docker build -t safe-ice .
   docker run --rm -it safe-ice

Development Installation
------------------------

For development, install additional dependencies:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git
   cd adaptive-importance-sampling-ice

   # Install with development dependencies
   poetry install --with dev

   # Install pre-commit hooks
   poetry run pre-commit install

   # Run tests
   poetry run pytest

Verifying Installation
----------------------

Check that Safe-ICE is installed correctly:

.. code-block:: python

   import safe_ice
   print(safe_ice.__version__)

   # Run a simple test
   from safe_ice import SafeICE
   import numpy as np

   def g(u):
       return 3.0 - np.linalg.norm(u, axis=-1)

   ice = SafeICE(g, dimension=2, N=100, max_iterations=2)
   pf, results = ice.run()
   print(f"Test successful! pf = {pf:.2e}")

Command-line interface:

.. code-block:: bash

   # Check CLI installation
   safe-ice --version

   # Run demo
   safe-ice demo

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'safe_ice'**

Make sure Safe-ICE is installed in your current environment:

.. code-block:: bash

   pip show safe-ice

**NumPy/SciPy version conflicts**

Update NumPy and SciPy:

.. code-block:: bash

   pip install --upgrade numpy scipy

**Poetry installation issues**

Clear Poetry cache and reinstall:

.. code-block:: bash

   poetry cache clear . --all
   poetry install

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**Windows**

On Windows, you might need Visual C++ redistributables for NumPy/SciPy:

1. Download from Microsoft's website
2. Install the x64 version
3. Restart your terminal

**macOS**

On Apple Silicon (M1/M2), use conda for best performance:

.. code-block:: bash

   conda install numpy scipy matplotlib
   pip install git+https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice.git

**Linux**

Ensure you have Python development headers:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install python3-dev

   # Fedora/RHEL
   sudo dnf install python3-devel

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/issues>`_
2. Read the :doc:`quickstart` guide
3. Ask on the discussions forum
4. Contact the maintainers

Next Steps
----------

* Follow the :doc:`quickstart` guide
* Explore :doc:`examples/index`
* Read about the :doc:`theory`
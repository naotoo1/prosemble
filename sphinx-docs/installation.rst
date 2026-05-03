Installation
============

Requirements
------------

- Python 3.11 or later
- JAX (with optional GPU support)

Install with pip
----------------

.. code-block:: bash

   pip install prosemble

Install with uv
----------------

.. code-block:: bash

   uv add prosemble

GPU Support
-----------

For GPU acceleration, install JAX with CUDA support first:

.. code-block:: bash

   # For CUDA 12
   pip install jax[cuda12]

   # Then install prosemble
   pip install prosemble

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/naotoo1/prosemble.git
   cd prosemble
   uv sync --all-extras

   # Or with devenv (Nix)
   devenv shell

Verify Installation
-------------------

.. code-block:: python

   >>> import prosemble
   >>> print(prosemble.__version__)
   1.0.0
   >>> import jax
   >>> print(jax.devices())  # Check available devices

Installation
============

Requirements
------------

- Python 3.11 or later
- JAX (installed automatically; GPU support is optional)

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

For GPU acceleration, install the CUDA 12 extra:

.. code-block:: bash

   pip install "prosemble[jax-cuda12]"

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
   2.4.0
   >>> import jax
   >>> print(jax.devices())  # Check available devices

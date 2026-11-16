Installation
============

Prosemble requires Python 3.11 or later. JAX, Optax and the other
dependencies are installed with it.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install prosemble

   .. tab-item:: uv

      .. code-block:: bash

         uv add prosemble

   .. tab-item:: GPU (CUDA 12)

      .. code-block:: bash

         pip install "prosemble[jax-cuda12]"

      This installs the CUDA build of JAX. See the
      `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
      for other accelerators and CUDA versions.

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/naotoo1/prosemble.git
         cd prosemble
         pip install -e .

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - Adds
   * - ``onnx``
     - ``onnx`` and ``onnxruntime``, for :doc:`ONNX export
       </user_guide/deployment>`.
   * - ``jax-cuda12``
     - The CUDA 12 build of JAX.
   * - ``docs``
     - The packages needed to build this documentation.
   * - ``dev``
     - Test, lint and release tools (see :doc:`/development/contributing`).

Install an extra with ``pip install "prosemble[onnx]"``.

Check the installation
----------------------

.. code-block:: python

   import jax
   import prosemble

   print(prosemble.__version__)
   print(jax.devices())     # CPU, or the GPUs JAX can use

If ``jax.devices()`` lists only the CPU on a machine with a GPU, the CPU
build of JAX is installed; install the GPU extra above.

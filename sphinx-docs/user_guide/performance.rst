Performance and scaling
=======================

Prosemble models are JAX programs, so they run unchanged on CPU, GPU and
TPU. This page covers compiled inference, accelerators, data-parallel
training on several devices, mixed precision and gradient checkpointing.

JIT-Compiled Inference
----------------------

Model inference (``predict``, ``predict_proba``) is automatically JIT-compiled
for maximum speed. The first call triggers compilation; subsequent calls
reuse the cached compilation.

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # First call: triggers JIT compilation
   preds = model.predict(X_test)

   # Subsequent calls: use cached compiled function (fast)
   preds = model.predict(X_test)

GPU Acceleration
----------------

JAX automatically uses available GPUs. Check your device:

.. code-block:: python

   import jax
   print(jax.devices())        # [GpuDevice(id=0)]
   print(jax.default_backend())  # 'gpu' or 'cpu'

All prosemble operations (training, inference, distance computation) run
on whatever device JAX is configured to use. No code changes needed.

.. _multi-device:

Multi-Device Data Parallelism
-----------------------------

For large datasets, prosemble supports data-parallel training across
several GPUs or TPUs with JAX's sharding API.

.. note::

   ``devices`` and ``gradient_checkpointing`` are currently accepted by
   :class:`~prosemble.models.GLVQ`, :class:`~prosemble.models.GLVQ1`,
   :class:`~prosemble.models.GLVQ21`, :class:`~prosemble.models.CELVQ`,
   :class:`~prosemble.models.LVQ1`, :class:`~prosemble.models.LVQ21`,
   :class:`~prosemble.models.MedianLVQ`, :class:`~prosemble.models.SLVQ`,
   :class:`~prosemble.models.RSLVQ`, the differentiating-kernel models
   :class:`~prosemble.models.DKGLVQ`, :class:`~prosemble.models.DKGRLVQ` and
   :class:`~prosemble.models.DKGMLVQ`, the one-class Neural Gas models and
   the one-class contrastive soft models. The other models, including the
   image and Siamese models, do not accept them yet.


.. code-block:: python

   import jax
   from prosemble.models import GLVQ

   # Use all available devices
   devices = jax.devices()  # e.g., [GpuDevice(id=0), GpuDevice(id=1), ...]

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       devices=devices,
       use_scan=False,
   )
   model.fit(X_train, y_train)

When ``devices`` is specified:

- Training data is **sharded** along the batch dimension across devices
- Model parameters and optimizer state are **replicated** on all devices
- Gradients are automatically aggregated (all-reduced) across devices
- After training, parameters are moved back to a single device for inference

**Requirements:**

- ``batch_size`` (if set) must be divisible by the number of devices
- The number of training samples should be divisible by the number of devices
- ``use_scan=False`` is recommended for multi-device training

``partial_fit()`` also works with multi-device models:

.. code-block:: python

   model.fit(X_batch1, y_batch1)
   model.partial_fit(X_batch2, y_batch2)  # data is automatically sharded

Mixed Precision
---------------

Supervised models support built-in mixed precision training via the
``mixed_precision`` parameter. Master weights stay in float32; the
forward/backward pass runs in lower precision for faster computation
and lower memory on GPU.

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.models import GLVQ

   # bfloat16 — recommended for training stability
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       mixed_precision='bfloat16',
   )
   model.fit(X_train, y_train)

   # float16 — maximum speed, uses loss scaling to prevent underflow
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       mixed_precision='float16',
   )
   model.fit(X_train, y_train)

   # Prototypes remain in float32
   assert model.prototypes_.dtype == jnp.float32

Gradient Checkpointing
----------------------

Gradient checkpointing (``jax.remat``) trades compute for memory by
recomputing forward activations during the backward pass instead of
storing them, which lowers memory use when the loss is expensive to
differentiate.

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       gradient_checkpointing=True,
   )
   model.fit(X_train, y_train)

When ``gradient_checkpointing=True``:

- Forward activations are **not** stored during the forward pass
- During backpropagation, the forward pass is **recomputed** to obtain
  activations needed for gradient computation
- Memory usage is reduced at the cost of ~33% more compute
- Results are numerically identical to standard training

For a shallow loss, such as GLVQ with the Euclidean distance, it saves
little. See the note under :ref:`multi-device` for the models that accept
this option.

Advanced Features
=================

Prosemble leverages JAX for GPU acceleration, JIT compilation, and
functional training loops. This guide covers advanced JAX-specific features.

JIT-Compiled Inference
-----------------------

Model inference (``predict``, ``predict_proba``) is automatically JIT-compiled
for maximum speed. The first call triggers compilation; subsequent calls
reuse the cached compilation.

.. code-block:: python

   import jax
   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # First call: triggers JIT compilation
   preds = model.predict(X_test)

   # Subsequent calls: use cached compiled function (fast)
   preds = model.predict(X_test)

``lax.scan`` Training
---------------------

Most models use ``jax.lax.scan`` for training loops, which compiles the
entire training loop into a single XLA program. This eliminates Python
loop overhead and enables efficient GPU execution.

Models that use ``lax.scan``:

- All fuzzy clustering models (FCM, PCM, FPCM, etc.)
- All supervised LVQ models (GLVQ, GRLVQ, etc.)
- Neural Gas, KohonenSOM, HeskesSOM

Models that use Python loops:

- Growing Neural Gas (dynamic topology)
- Riemannian Neural Gas (manifold operations)
- Median LVQ (combinatorial M-step)

Mixed Precision
---------------

Train with reduced precision for faster computation and lower memory usage:

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.models import FCM

   # Convert data to float16
   X_f16 = X_train.astype(jnp.float16)

   model = FCM(n_clusters=3, fuzzifier=2.0)
   model.fit(X_f16)

   # Or use bfloat16 (better for training stability)
   X_bf16 = X_train.astype(jnp.bfloat16)
   model.fit(X_bf16)

Mini-Batch Training
-------------------

For large datasets, train on random mini-batches each iteration:

.. code-block:: python

   from prosemble.models import FCM

   model = FCM(
       n_clusters=3,
       fuzzifier=2.0,
       max_iter=200,
       batch_size=64,        # mini-batch size
       random_seed=42,
   )
   model.fit(X_large)

Model Quantization
------------------

Reduce model size for deployment by quantizing prototypes and parameters:

.. code-block:: python

   from prosemble.core.quantization import quantize_model

   model.fit(X_train, y_train)

   # Quantize to int8 (smallest)
   quantized = quantize_model(model, dtype='int8')

   # Quantize to float16 (balanced)
   quantized = quantize_model(model, dtype='float16')

Checkpointing and Resume
-------------------------

Save and load models for persistence:

.. code-block:: python

   # Save
   model.save('my_model.npz')

   # Load
   from prosemble.models import GLVQ
   loaded = GLVQ.load('my_model.npz')

   # Resume training from checkpoint
   loaded.fit(X_train, y_train, resume=True, max_iter=50)

Early Stopping with Patience
------------------------------

Fuzzy clustering models support patience-based early stopping:

.. code-block:: python

   from prosemble.models import FCM

   model = FCM(
       n_clusters=3,
       max_iter=500,
       epsilon=1e-5,        # convergence tolerance
   )
   model.fit(X)

   # Training stops when centroid change < epsilon
   print(f"Stopped at iteration {model.n_iter_}")

GPU Acceleration
----------------

JAX automatically uses available GPUs. Check your device:

.. code-block:: python

   import jax
   print(jax.devices())        # [GpuDevice(id=0)]
   print(jax.default_backend())  # 'gpu' or 'cpu'

All prosemble operations (training, inference, distance computation) run
on whatever device JAX is configured to use. No code changes needed.

Training Callbacks
------------------

Monitor training with callbacks:

.. code-block:: python

   # Training history is automatically tracked
   model.fit(X_train, y_train)

   # Access history
   print(model.loss_history_)          # supervised models
   print(model.objective_history_)     # clustering models

Live Visualization
------------------

Enable real-time training visualization for clustering models:

.. code-block:: python

   from prosemble.models import FCM

   model = FCM(
       n_clusters=3,
       max_iter=100,
       plot_steps=True,      # enable live visualization
   )
   model.fit(X)

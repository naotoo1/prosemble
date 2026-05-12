Advanced Features
=================

Prosemble leverages JAX for GPU acceleration, JIT compilation, and
functional training loops. This guide covers all advanced features
available for supervised prototype models and fuzzy clustering models.

JIT-Compiled Inference
-----------------------

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

Training Loop Modes
--------------------

``lax.scan`` vs Python Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, supervised models use ``jax.lax.scan`` which compiles the
entire training loop into a single XLA program. This eliminates Python
loop overhead and is the fastest option.

Set ``use_scan=False`` to switch to a Python loop, which is required
when using callbacks, patience-based early stopping, or validation
monitoring.

.. code-block:: python

   from prosemble.models import GLVQ

   # Default: lax.scan (fastest, no callbacks)
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       use_scan=True,       # default
   )
   model.fit(X_train, y_train)

   # Python loop: needed for callbacks, patience, validation
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       use_scan=False,
   )
   model.fit(X_train, y_train)

Models that always use Python loops regardless of ``use_scan``:

- Growing Neural Gas (dynamic topology)
- Riemannian Neural Gas (manifold operations)
- Median LVQ (combinatorial M-step)

Fuzzy clustering models use ``lax.scan`` by default and fall back to a
Python loop when callbacks, patience, or ``restore_best`` are enabled.

Mini-Batch Training
^^^^^^^^^^^^^^^^^^^^

For large datasets, supervised models support mini-batch training.
Each iteration samples a random batch from the training data:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       batch_size=64,
   )
   model.fit(X_large, y_large)

Online / Incremental Learning
------------------------------

Train on streaming data or update an existing model with new batches
using ``partial_fit()``. The model must be fitted first via ``fit()``,
then each ``partial_fit()`` call performs a single gradient update while
preserving optimizer state across calls.

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)

   # Initial training
   model.fit(X_train, y_train)

   # Incremental updates with new data batches
   for X_batch, y_batch in data_stream:
       model.partial_fit(X_batch, y_batch)

   # Prototypes and optimizer state are preserved across calls
   preds = model.predict(X_test)

All supervised models support ``partial_fit()``.

Sample Weighting and Class Balancing
-------------------------------------

Per-Sample Weights
^^^^^^^^^^^^^^^^^^^

Pass ``sample_weight`` to ``fit()`` to assign different importance to
individual training samples. Samples with higher weights have more
influence on the loss:

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)

   # Per-sample weights (e.g., higher weight for difficult samples)
   weights = jnp.array([1.0, 1.0, 2.0, 1.0, 3.0, ...])
   model.fit(X_train, y_train, sample_weight=weights)

Class Balancing
^^^^^^^^^^^^^^^^

For imbalanced datasets, use ``class_weight`` to automatically compute
per-sample weights inversely proportional to class frequency:

.. code-block:: python

   from prosemble.models import GLVQ

   # Automatic balanced weighting
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       class_weight='balanced',
   )
   model.fit(X_train, y_train)

   # Or specify weights per class manually
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       class_weight={0: 1.0, 1: 5.0, 2: 1.0},
   )
   model.fit(X_train, y_train)

With ``class_weight='balanced'``, the weight for class *c* is computed as:

.. math::

   w_c = \frac{N}{K \cdot N_c}

where *N* is the total number of samples, *K* is the number of classes,
and *N_c* is the number of samples in class *c*.

Validation and Early Stopping
------------------------------

Validation Monitoring
^^^^^^^^^^^^^^^^^^^^^^

Pass validation data to ``fit()`` to monitor performance on held-out
samples during training. Combined with ``restore_best=True``, the model
automatically restores the parameters that achieved the lowest
validation loss:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       restore_best=True,
       use_scan=False,      # required for validation monitoring
   )
   model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
   )

   # model.prototypes_ now holds the best params from training
   print(f"Best validation loss: {model.best_loss_}")

Patience-Based Early Stopping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stop training early when no improvement is observed for a given
number of consecutive iterations:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=500,
       lr=0.01,
       patience=20,
       restore_best=True,
       use_scan=False,
   )
   model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
   )

   print(f"Stopped at iteration {model.n_iter_}")

Fuzzy clustering models also support patience-based early stopping:

.. code-block:: python

   from prosemble.models import FCM

   model = FCM(
       n_clusters=3,
       max_iter=500,
       patience=10,
       restore_best=True,
   )
   model.fit(X)

   print(f"Stopped at iteration {model.n_iter_}")

Optimizer Configuration
------------------------

Prosemble supports 26 optimizers via optax. Pass a string name or a
pre-built ``optax.GradientTransformation`` to the ``optimizer`` parameter:

.. code-block:: python

   from prosemble.models import GLVQ

   # String name (26 options)
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       optimizer='adamw',
   )
   model.fit(X_train, y_train)

Available optimizer strings: ``adam``, ``adamw``, ``adamax``, ``adamaxw``,
``adan``, ``adabelief``, ``amsgrad``, ``radam``, ``lamb``, ``lion``,
``novograd``, ``sgd``, ``sign_sgd``, ``signum``, ``noisy_sgd``, ``lars``,
``rmsprop``, ``adagrad``, ``adadelta``, ``adafactor``, ``sm3``, ``yogi``,
``rprop``, ``fromage``, ``lbfgs``, ``dpsgd``.

You can also pass a custom optax optimizer directly:

.. code-block:: python

   import optax

   custom_opt = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.adam(learning_rate=0.001),
   )

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       optimizer=custom_opt,
   )
   model.fit(X_train, y_train)

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^

Decay or warm up the learning rate during training using the
``lr_scheduler`` parameter. Nine built-in schedules are available:

.. code-block:: python

   from prosemble.models import GLVQ

   # Cosine decay: lr decays smoothly from 0.01 to 0 over training
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       lr_scheduler='cosine_decay',
   )
   model.fit(X_train, y_train)

   # Exponential decay with custom decay rate
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       lr_scheduler='exponential_decay',
       lr_scheduler_kwargs={'decay_rate': 0.95, 'transition_steps': 1},
   )
   model.fit(X_train, y_train)

   # Warmup then cosine decay
   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       lr_scheduler='warmup_cosine_decay',
       lr_scheduler_kwargs={
           'warmup_steps': 20,
           'peak_value': 0.01,
           'end_value': 0.0,
       },
   )
   model.fit(X_train, y_train)

Available schedules:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Schedule
     - Description
   * - ``exponential_decay``
     - Exponential decay: ``lr * decay_rate^(step / transition_steps)``
   * - ``cosine_decay``
     - Cosine annealing from ``lr`` to 0
   * - ``warmup_cosine_decay``
     - Linear warmup then cosine decay
   * - ``warmup_exponential_decay``
     - Linear warmup then exponential decay
   * - ``warmup_constant``
     - Linear warmup then constant learning rate
   * - ``polynomial``
     - Polynomial decay with configurable power
   * - ``linear``
     - Linear decay from ``lr`` to ``end_value``
   * - ``piecewise_constant``
     - Step-wise constant schedule with boundaries
   * - ``sgdr``
     - Cosine annealing with warm restarts (SGDR)

Lookahead Optimizer
^^^^^^^^^^^^^^^^^^^^

Lookahead maintains two sets of weights — "fast" weights updated every
step, and "slow" weights updated by interpolating toward the fast weights
every *k* steps. This improves generalization and reduces variance:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       lookahead={
           'sync_period': 6,         # sync slow weights every 6 steps
           'slow_step_size': 0.5,    # interpolation factor
       },
       use_scan=False,               # required for lookahead
   )
   model.fit(X_train, y_train)

After training, the slow weights (which generalize better) are used for
inference.

Gradient Accumulation
^^^^^^^^^^^^^^^^^^^^^^

Accumulate gradients over multiple steps before applying an update,
effectively increasing the batch size without increasing memory usage:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       batch_size=16,
       gradient_accumulation_steps=4,  # effective batch size = 16 * 4 = 64
   )
   model.fit(X_train, y_train)

Parameter Freezing
^^^^^^^^^^^^^^^^^^^

Freeze specific parameters during training. Frozen parameters receive
zero gradients and remain at their initial values:

.. code-block:: python

   from prosemble.models import GMLVQ

   model = GMLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       freeze_params=['omega'],       # freeze the metric matrix
   )
   model.fit(X_train, y_train)

   # Only prototypes were updated; omega stayed at its initial value

Common use cases:

- Freeze ``'omega'`` in GMLVQ to train prototypes with a fixed metric
- Freeze ``'prototypes'`` to learn only the metric adaptation matrix
- Two-phase training: first train prototypes, then freeze them and train omega

Exponential Moving Average
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maintain an exponential moving average of parameters during training.
EMA parameters often generalize better than the final training
parameters:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       ema_decay=0.999,
       use_scan=False,
   )
   model.fit(X_train, y_train)

The EMA update rule at each step is:

.. math::

   \theta_{\text{ema}} = \alpha \cdot \theta_{\text{ema}} + (1 - \alpha) \cdot \theta

where :math:`\alpha` is the ``ema_decay`` factor.

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

Checkpointing and Resume
-------------------------

Save and load models for persistence. All fitted parameters, optimizer
state, and hyperparameters are preserved:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # Save
   model.save('my_model.npz')

   # Load
   loaded = GLVQ.load('my_model.npz')

   # Resume training from checkpoint
   loaded.fit(X_train, y_train, resume=True)

Model Quantization
------------------

Reduce model size for deployment by quantizing prototypes and parameters
when saving. The model in memory is unchanged; only the saved file is
quantized:

.. code-block:: python

   model.fit(X_train, y_train)

   # Quantize to int8 on save (smallest file size)
   model.save('model_int8.npz', quantize='int8')

   # Quantize to float16 on save (balanced)
   model.save('model_f16.npz', quantize='float16')

   # Load quantized model
   loaded = GLVQ.load('model_int8.npz')

Export for Deployment
----------------------

Export a JIT-compiled prediction function using ``jax.export`` for
deployment without the full model or prosemble dependency:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # Export compiled predict function for batch size 32
   exported = model.export_predict(batch_size=32)

   # Serialize to bytes
   blob = exported.serialize()

   # Later, in a separate process without prosemble:
   import jax
   loaded = jax.export.deserialize(blob)
   preds = loaded.call(X_batch)

The exported function contains only the compiled XLA computation and
the frozen prototype/metric parameters — no Python dependencies needed
at inference time.

Prototype Analysis
-------------------

Prototype Win Ratios
^^^^^^^^^^^^^^^^^^^^^

Analyze how often each prototype wins on correctly classified samples.
This helps identify "dead" prototypes that never win and may be
candidates for removal or reinitialization:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=3, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   ratios = model.prototype_win_ratios(X_train, y_train)

   for i, r in enumerate(ratios):
       label = model.prototype_labels_[i]
       print(f"Prototype {i} (class {label}): win ratio {r:.3f}")

A win ratio of 0.0 means the prototype never won on any correctly
classified sample and is effectively unused.

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

Monitor training with callbacks. Training history is automatically
tracked for all models:

.. code-block:: python

   model.fit(X_train, y_train)

   # Supervised models
   print(model.loss_history_)

   # Clustering models
   print(model.objective_history_)

Live Visualization
------------------

Enable real-time training visualization for clustering models:

.. code-block:: python

   from prosemble.models import FCM

   model = FCM(
       n_clusters=3,
       max_iter=100,
       plot_steps=True,
   )
   model.fit(X)

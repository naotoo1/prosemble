Training
========

Every model is trained by ``fit``. This page covers the options that
control how: the training loop, mini-batches, sample and class weights,
validation and early stopping, optimizers and learning-rate schedules,
regularization, curriculum learning and custom gradients. Unless noted,
the options apply to all supervised models; most also apply to the
clustering models.

.. tip::

   By default the whole training loop is compiled into one XLA program
   with ``jax.lax.scan``, which runs all ``max_iter`` iterations. Options
   that need Python control flow (validation data, patience, class or
   sample weights, EMA, lookahead) switch to a Python loop automatically.

Training Loop Modes
-------------------

``lax.scan`` vs Python Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

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
-----------------------------

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
------------------------------------

Per-Sample Weights
~~~~~~~~~~~~~~~~~~

Pass ``sample_weight`` to ``fit()`` to assign different importance to
individual training samples. Samples with higher weights have more
influence on the loss:

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)

   # Per-sample weights, here three times the weight for the first 10 samples
   weights = jnp.ones(len(y_train)).at[:10].set(3.0)
   model.fit(X_train, y_train, sample_weight=weights)

Class Balancing
~~~~~~~~~~~~~~~

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
-----------------------------

Validation Monitoring
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
-----------------------

Prosemble supports 25 optimizers via optax. Pass a string name or a
pre-built ``optax.GradientTransformation`` to the ``optimizer`` parameter:

.. code-block:: python

   from prosemble.models import GLVQ

   # String name (25 options)
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
``rprop``, ``fromage``, ``lbfgs``.

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
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Custom LVQ Optimizers
---------------------

Prosemble provides three custom optax-compatible optimizers designed
specifically for the geometry and parameter structure of LVQ models.
These can be composed with standard optax transformations via
``optax.chain()``.

Per-Group Gradient Clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different parameter types (prototypes, omega matrices, relevances) have
different natural scales. A single global clip either under-constrains
large parameters or over-constrains small ones. Per-group clipping clips
each parameter group independently:

.. code-block:: python

   import optax
   from prosemble.core.optimizers import per_group_clip

   optimizer = optax.chain(
       per_group_clip({'prototypes': 1.0, 'omega': 0.5, 'sigmas': 0.1}),
       optax.adam(0.01),
   )

   from prosemble.models import GMLVQ
   model = GMLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       optimizer=optimizer,
   )
   model.fit(X_train, y_train)

Hypergradient Descent
~~~~~~~~~~~~~~~~~~~~~

Adaptive per-parameter learning rates via gradient correlation (Baydin
et al. 2017). If consecutive gradients point in the same direction,
the learning rate increases; if they oscillate, it decreases:

.. math::

   \eta_k^{t+1} = \text{clip}\left(
       \eta_k^t + \beta \cdot \langle g_k^t, g_k^{t-1} \rangle
   \right)

.. code-block:: python

   from prosemble.core.optimizers import hypergradient_descent
   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       optimizer=hypergradient_descent(
           init_lr=0.01,
           hyper_lr=1e-4,
           min_lr=1e-6,
           max_lr=0.1,
       ),
   )
   model.fit(X_train, y_train)

Riemannian Nesterov Momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nesterov accelerated gradient providing :math:`O(1/t^2)` convergence
rate versus :math:`O(1/t)` for vanilla gradient descent. Designed for
use with Riemannian models where manifold projection is handled by
``_post_update()``:

.. code-block:: python

   from prosemble.core.optimizers import riemannian_nesterov
   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       optimizer=riemannian_nesterov(
           learning_rate=0.01,
           momentum=0.9,
       ),
   )
   model.fit(X_train, y_train)

Prototype Regularization
------------------------

Prototype Diversity (DPP)
~~~~~~~~~~~~~~~~~~~~~~~~~

Prevent same-class prototypes from collapsing onto each other using a
determinantal point process (DPP) inspired penalty. The log-determinant
of the distance kernel matrix encourages spread:

.. code-block:: python

   from prosemble.core.regularization import prototype_diversity_loss

   # Add as a regularization term to the loss
   div_loss = prototype_diversity_loss(
       prototypes, proto_labels,
       sigma_div=1.0,
   )
   total_loss = classification_loss + 0.01 * div_loss

Sparse Relevance Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For relevance learning models (GRLVQ), encourage sparsity in the
learned relevance profile via proximal gradient operators:

.. code-block:: python

   from prosemble.core.regularization import (
       sparse_relevance_proximal,
       elastic_net_proximal,
   )

   # L1 soft-thresholding (promotes exact zeros)
   sparse_rel = sparse_relevance_proximal(
       relevances, l1_weight=0.01, lr=0.01,
   )

   # Elastic net (L1 + L2 combination)
   sparse_rel = elastic_net_proximal(
       relevances, l1_weight=0.01, l2_weight=0.001, lr=0.01,
   )

Curriculum Learning
-------------------

Self-paced curriculum learning adaptively weights training samples by
difficulty. Early in training, the model focuses on easy samples; as
training progresses, harder samples are gradually introduced.

The difficulty measure is the per-sample GLVQ loss (mu-ratio). Samples
with loss below the current threshold are weighted; those above are
down-weighted or excluded.

.. code-block:: python

   from prosemble.core.curriculum import (
       curriculum_weights, curriculum_threshold, apply_curriculum_to_loss,
   )

   # Compute per-sample difficulty weights
   threshold = curriculum_threshold(
       iteration=50,
       max_iter=200,
       init_threshold=0.3,   # initially only easy samples
       final_threshold=1.0,  # eventually all samples
       schedule='linear',
   )
   weights = curriculum_weights(per_sample_loss, threshold, mode='soft')

   # Or use the combined pipeline
   weighted_loss = apply_curriculum_to_loss(
       per_sample_losses,
       iteration=50,
       max_iter=200,
       init_threshold=0.3,
       final_threshold=1.0,
   )

Three weighting modes are available:

- ``'hard'``: binary — include (weight=1) or exclude (weight=0)
- ``'soft'``: smooth sigmoid transition at the threshold
- ``'linear'``: linearly decrease weight from 1 to 0 as loss increases

Three scheduling strategies:

- ``'linear'``: threshold increases linearly with iteration
- ``'exponential'``: slow start, fast growth
- ``'cosine'``: smooth cosine schedule from init to final threshold

Training Callbacks
------------------

Monitor training with callbacks. Training history is automatically
tracked for all models:

.. code-block:: python

   model.fit(X_train, y_train)
   print(model.loss_history_)          # supervised models

   from prosemble.models import FCM

   fcm = FCM(n_clusters=3).fit(X_train)
   print(fcm.objective_history_)       # clustering models

Custom Gradient Rules
---------------------

Prosemble provides infrastructure for subclasses to define custom
backward passes using ``jax.custom_vjp``. This is useful for:

- Optimal transport distances with non-differentiable sorting operations
- Hard thresholding operations that need surrogate gradients
- Distances with numerically unstable standard autodiff

Subclasses opt in by setting ``self._use_custom_vjp = True`` in their
``__init__`` and overriding the ``_custom_vjp_loss`` method:

.. code-block:: python

   class MyModel(GLVQ):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self._use_custom_vjp = True

       def _custom_vjp_loss(self, params, X, y, proto_labels):
           # Implement custom forward + backward pass
           ...

When ``_use_custom_vjp`` is active, gradient checkpointing is
automatically disabled (custom VJP already controls what gets
saved/recomputed).

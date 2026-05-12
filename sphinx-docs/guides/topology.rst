Topology-Preserving Models
==========================

Topology-preserving models learn the structure of unlabeled data by
positioning prototypes such that neighborhood relationships are preserved.

Neural Gas
----------

Neural Gas distributes prototypes to match data density using **rank-based
neighborhood cooperation**. No predefined grid structure is needed.

For each sample, prototypes are ranked by distance. The cooperation weight
for rank :math:`k` is :math:`h(k) = \exp(-k / \lambda)`, where :math:`\lambda`
controls the neighborhood range.

.. code-block:: python

   from prosemble.models import NeuralGas
   from prosemble.datasets import load_iris_jax

   dataset = load_iris_jax()
   X = dataset.input_data

   model = NeuralGas(
       n_prototypes=10,
       max_iter=100,
       lr_init=0.5,          # initial learning rate
       lr_final=0.01,        # final learning rate
       lambda_init=5.0,      # initial neighborhood range
       lambda_final=0.01,    # final (narrow) range
       random_seed=42,
   )
   model.fit(X)

   # Assign samples to nearest prototype
   labels = model.predict(X)

   # Distance to all prototypes
   distances = model.transform(X)

   print(f"Prototypes: {model.prototypes_.shape}")
   print(f"Iterations: {model.n_iter_}")
   print(f"Energy: {model.loss_:.4f}")

**How the ranking works:**

1. Compute distances from sample :math:`x` to all prototypes
2. Sort prototypes by distance: nearest = rank 0, farthest = rank K-1
3. Cooperation weight :math:`h(k) = \exp(-k / \lambda)`
4. Update all prototypes: :math:`w_j \leftarrow w_j + \varepsilon \cdot h(k_j) \cdot (x - w_j)`

Large :math:`\lambda` means many prototypes cooperate (global ordering).
Small :math:`\lambda` means only nearest prototypes update (local refinement).

Kohonen SOM
------------

Self-Organizing Map arranges prototypes on a fixed 2D grid. Neighborhoods
are defined by **grid position**, not data-space distance.

.. code-block:: python

   from prosemble.models import KohonenSOM

   model = KohonenSOM(
       grid_height=5,
       grid_width=5,
       sigma_init=2.0,       # initial grid neighborhood width
       sigma_final=0.5,      # final (narrow)
       lr_init=0.5,
       lr_final=0.01,
       max_iter=100,
       random_seed=42,
   )
   model.fit(X)

   labels = model.predict(X)
   print(f"Prototypes: {model.prototypes_.shape}")  # (25, n_features)

The key difference from Neural Gas: SOM neighborhoods are defined by fixed
grid distance, so the 2D grid topology is preserved in the mapping. This
makes SOM ideal for data visualization on a 2D map.

Heskes SOM
----------

Heskes SOM modifies the BMU selection criterion. Instead of choosing the
nearest prototype, it selects the unit whose **entire neighborhood** best
represents the sample:

:math:`c^* = \arg\min_c \sum_k h(k, c) \cdot \|x - w_k\|^2`

This has a well-defined energy function (unlike standard Kohonen SOM),
guaranteeing convergence.

.. code-block:: python

   from prosemble.models import HeskesSOM

   model = HeskesSOM(
       grid_height=5,
       grid_width=5,
       sigma_init=2.0,
       sigma_final=0.5,
       max_iter=100,
       random_seed=42,
   )
   model.fit(X)

Growing Neural Gas
------------------

Starts with 2 nodes and dynamically **adds and removes** prototypes during
training, adapting model complexity to the data.

.. code-block:: python

   from prosemble.models import GrowingNeuralGas

   model = GrowingNeuralGas(
       max_nodes=30,
       max_iter=100,
       lr_winner=0.1,
       lr_neighbor=0.01,
       max_age=50,            # remove edges older than this
       insert_interval=100,   # insert new node every N steps
       error_decay=0.995,
       random_seed=42,
   )
   model.fit(X)

   print(f"Final nodes: {model.prototypes_.shape[0]}")  # may be < max_nodes

**Node insertion** places new prototypes where the representation error
is highest, automatically focusing model capacity on complex data regions.

.. note::

   Growing Neural Gas is **not JIT-compilable** due to its dynamic
   topology (changing number of nodes and edges). It uses Python loops
   instead of ``lax.scan``.

.. note::

   **Riemannian Neural Gas** (Neural Gas on manifolds: SO(n), SPD(n),
   Grassmannian) is available via ``prosemble.models.RiemannianNeuralGas``.

Choosing a Model
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Model
     - Topology
     - Best For
   * - Neural Gas
     - Rank-based (adaptive)
     - General data structure learning
   * - Kohonen SOM
     - Fixed 2D grid
     - Visualization on a 2D map
   * - Heskes SOM
     - Fixed 2D grid
     - Principled SOM with convergence guarantees
   * - Growing NG
     - Dynamic edges
     - Unknown number of clusters

Learning the shape of data
==========================

.. currentmodule:: prosemble.models

Topology-preserving models place prototypes so that neighbouring
prototypes cover neighbouring parts of the data. The result is a graph or
a grid that summarises the shape of the data with a few dozen points.

Growing Neural Gas
------------------

:class:`GrowingNeuralGas` starts with two prototypes and adds new ones
where the representation error is largest, connecting prototypes that are
often the two nearest to a sample. On a noisy spiral it recovers the curve
as a chain of prototypes:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from prosemble.models import GrowingNeuralGas

   rng = np.random.default_rng(2)
   t = rng.uniform(0.5, 3.6 * np.pi, 900)
   spiral = np.c_[t * np.cos(t), t * np.sin(t)] / 4 + rng.normal(0, 0.12, (900, 2))
   spiral = jnp.asarray(spiral, dtype=jnp.float32)

   gng = GrowingNeuralGas(max_nodes=60, max_iter=60, random_seed=0)
   gng.fit(spiral)

   gng.prototypes_      # node positions
   gng.edges_           # (60, 60) adjacency of the learned graph

Self-organizing map
-------------------

:class:`KohonenSOM` arranges its prototypes on a fixed grid and pulls grid
neighbours towards each other. The grid unfolds over the data, so nearby
grid nodes represent nearby data, which is what makes SOMs useful for
visualising high-dimensional data on a 2-D grid:

.. code-block:: python

   from prosemble.models import KohonenSOM

   square = jnp.asarray(rng.uniform(-1, 1, (900, 2)), dtype=jnp.float32)
   som = KohonenSOM(grid_height=12, grid_width=12, max_iter=200,
                    random_seed=0)
   som.fit(square)

   grid = som.prototypes_.reshape(12, 12, 2)

.. figure:: /_static/figures/tutorial-topology-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-topology-dark.png
   :class: only-dark pr-figure

   Left: the graph learned by Growing Neural Gas on a spiral. Right: a
   12 × 12 self-organizing map unfolded over a square.

Next steps
----------

:doc:`/user_guide/topology` covers Neural Gas, the Heskes SOM and Neural
Gas on Riemannian manifolds.

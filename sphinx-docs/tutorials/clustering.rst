Fuzzy clustering
================

.. currentmodule:: prosemble.models

Hard clustering puts every sample in exactly one cluster. Fuzzy c-means
(:class:`FCM`) instead gives every sample a degree of membership in every
cluster, which shows which samples sit between clusters.

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from prosemble.models import FCM

   rng = np.random.default_rng(1)
   centres = np.array([[0, 0], [3.2, 0.6], [1.4, 2.9]])
   X = jnp.asarray(np.concatenate([rng.normal(c, 0.65, (120, 2)) for c in centres]),
                   dtype=jnp.float32)

   model = FCM(n_clusters=3, fuzzifier=2.0, max_iter=200, random_seed=0)
   model.fit(X)

   model.centroids_                 # (3, 2) cluster centres
   memberships = model.predict_proba(X)   # (360, 3), rows sum to 1
   labels = model.predict(X)              # cluster of largest membership

.. figure:: /_static/figures/tutorial-clustering-light.png
   :class: only-light pr-figure
   :width: 75%
   :align: center

.. figure:: /_static/figures/tutorial-clustering-dark.png
   :class: only-dark pr-figure
   :width: 75%
   :align: center

   Colours mix the three cluster colours in proportion to the memberships;
   pale regions are shared between clusters. Large markers are the
   centroids.

Most samples belong clearly to one cluster, but 6 % have no membership
above 0.6. These are the samples between clusters, where a hard
assignment would hide the ambiguity.

The fuzzifier :math:`m` (``fuzzifier``) sets how soft the memberships are:
as :math:`m \to 1` FCM approaches hard c-means, and larger values spread
the memberships more evenly.

Next steps
----------

* Outliers pull FCM's centroids, because every sample's memberships must
  sum to one. :class:`PCM` and :class:`PFCM` add possibilistic
  memberships ("typicality") that are robust to them.
* Kernel variants (:class:`KFCM` and others) find clusters that are not
  round. See :doc:`/user_guide/clustering`.

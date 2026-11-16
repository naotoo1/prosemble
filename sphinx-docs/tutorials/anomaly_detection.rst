Anomaly detection
=================

.. currentmodule:: prosemble.models

Often only normal data are available: readings from a machine that works,
transactions known to be genuine. A one-class model learns what normal
looks like and flags whatever does not fit. This tutorial fits one to
normal samples alone and tests it on outliers it has never seen.

The data
--------

Normal samples lie on a broken ring and in a separate blob. Outliers for
testing are drawn uniformly, away from the normal data:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp

   rng = np.random.default_rng(0)
   a = rng.uniform(0, 1.5 * np.pi, 300)
   ring = np.c_[2 * np.cos(a), 2 * np.sin(a)] + rng.normal(0, 0.25, (300, 2))
   blob = rng.normal([3.2, -2.2], 0.35, (100, 2))
   normal = jnp.asarray(np.r_[ring, blob], dtype=jnp.float32)

Train on normal samples only
----------------------------

:class:`OCDKRSLVQ` models the density of the normal class with Gaussian
kernel prototypes. It learns it by telling the normal samples apart from
samples of a broad reference distribution that it draws itself, so no
outliers are needed for training:

.. code-block:: python

   from prosemble.models import OCDKRSLVQ

   model = OCDKRSLVQ(n_prototypes=8, max_iter=300, lr=0.05, random_seed=0)
   model.fit(normal)

   X_new = jnp.array([[0.0, 2.0],     # on the ring
                      [0.0, 0.0]])    # in its centre
   scores = model.decision_function(X_new)    # in (0, 1); 0.5 is the boundary
   labels = model.predict(X_new)              # 0 = normal, 1 = outlier

.. figure:: /_static/figures/tutorial-anomaly-light.png
   :class: only-light pr-figure
   :width: 75%
   :align: center

.. figure:: /_static/figures/tutorial-anomaly-dark.png
   :class: only-dark pr-figure
   :width: 75%
   :align: center

   The learned boundary (line) and score (shading) around the normal data,
   with the eight prototypes. Crosses are new outliers.

The prototypes spread along the ring and into the blob, and the boundary
follows the shape of the normal data, including the gap in the ring. The
model accepts 95 % of the normal samples, as set by its default
``nu=0.05`` (the share of training samples allowed outside the boundary),
and flags all 40 test outliers.

Which one-class model?
----------------------

Models with learned visibility thresholds, such as :class:`OCGLVQ`, set
each threshold by trading off normal samples against outliers, so they
need some labelled outliers in the training data; fitted to normal data
alone, they accept everything. The contrastive soft models
(:class:`OCDKRSLVQ`, :class:`OCWRSLVQ`, :class:`OCRiemannianRSLVQ`) can be
fitted to normal data alone and, when a few labelled outliers are
available, give them prototypes of their own. See
:doc:`/user_guide/one_class`.

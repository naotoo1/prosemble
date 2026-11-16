Probabilistic classifiers
=========================

.. currentmodule:: prosemble.models

Probabilistic LVQ models treat each prototype as the centre of a Gaussian
component of a class-conditional mixture. A sample's class probability is
the share of the mixture density that belongs to that class, so
``predict_proba`` returns posterior probabilities rather than scores
derived from distances. Robust soft LVQ (:class:`RSLVQ`) maximises the
likelihood ratio of the correct class,

.. math::

   \ell(x, y) = \log \frac{\sum_{k:\, c(k) = y} \exp f_k(x)}
                          {\sum_{k} \exp f_k(x)},
   \qquad f_k(x) = -\frac{d(x, w_k)}{2\sigma^2},

where :math:`c(k)` is the class of prototype :math:`k`. In :class:`RSLVQ`
and its matrix variants the width :math:`\sigma` is fixed; the
kernel, Wasserstein and Riemannian soft models can also learn it
(``adapt_sigma=True``).

The same cost is available with other distances:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Distance
     - Models
   * - Euclidean, learned matrix
     - :class:`RSLVQ`, :class:`MRSLVQ`, :class:`LMRSLVQ`, :class:`SLVQ`
   * - With Neural Gas cooperation
     - :class:`RSLVQ_NG`, :class:`MRSLVQ_NG`, :class:`LMRSLVQ_NG`
       (:doc:`neural_gas`)
   * - Differentiating kernels
     - :class:`DKRSLVQ`, :class:`DKMRSLVQ`, :class:`DKLMRSLVQ`
       (:doc:`/user_guide/kernels`)
   * - 2-Wasserstein (Gaussian prototypes)
     - :class:`WRSLVQ`, :class:`WMRSLVQ`, :class:`WLMRSLVQ`
       (:doc:`wasserstein`)
   * - Riemannian manifolds
     - :class:`RiemannianRSLVQ`, :class:`RiemannianMRSLVQ`
       (:doc:`/user_guide/riemannian`)

All of them can be calibrated and can reject uncertain samples; see
:doc:`/user_guide/probabilities`.

Probabilistic Models (SLVQ, RSLVQ)
----------------------------------

Treat prototypes as Gaussian components. RSLVQ provides robust
log-likelihood training with confidence-based rejection.

.. code-block:: python

   from prosemble.models import RSLVQ

   model = RSLVQ(
       n_prototypes_per_class=2,
       sigma=1.0,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

**MRSLVQ** — Matrix RSLVQ with global Omega metric learning:

.. code-block:: python

   from prosemble.models import MRSLVQ

   model = MRSLVQ(
       n_prototypes_per_class=1,
       latent_dim=2,
       sigma=1.0,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)
   print(model.omega_matrix.shape)   # (n_features, latent_dim)
   print(model.lambda_matrix)        # Omega^T @ Omega

**LMRSLVQ** — Local Matrix RSLVQ with per-prototype Omega matrices:

.. code-block:: python

   from prosemble.models import LMRSLVQ

   model = LMRSLVQ(
       n_prototypes_per_class=1,
       latent_dim=2,
       sigma=1.0,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

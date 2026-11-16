Distributional prototypes
=========================

.. currentmodule:: prosemble.models

The Wasserstein LVQ family uses Gaussian distributional prototypes instead
of point prototypes.  Each prototype :math:`k` is a diagonal Gaussian
:math:`\mathcal{N}(\mu_k, \text{diag}(\sigma_k^2))`.  The distance from
a data point :math:`x` (treated as a Dirac delta) to prototype :math:`k`
is the squared 2-Wasserstein distance:

.. math::

   W_2^2(\delta_x, \mathcal{N}(\mu_k, \text{diag}(\sigma_k^2)))
   = \|x - \mu_k\|^2 + \sum_j \sigma_{kj}^2

The variance term acts as a per-prototype learned "spread" — prototypes
with smaller variance are more specific and attract nearby points more
strongly.

**WGLVQ** — Wasserstein GLVQ (base variant):

.. code-block:: python

   from prosemble.models import WGLVQ

   model = WGLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Learned prototype means and variances
   print(model.prototype_means_.shape)      # (n_prototypes, n_features)
   print(model.prototype_variances_.shape)  # (n_prototypes, n_features)

**WGMLVQ** — Adds a global :math:`\Omega` matrix projection:

.. math::

   W_2^2(x, k) = \|\Omega(x - \mu_k)\|^2 + \sum_j \sigma_{kj}^2

.. code-block:: python

   from prosemble.models import WGMLVQ

   model = WGMLVQ(
       n_prototypes_per_class=2,
       latent_dim=2,          # project to 2D
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

   print(model.omega_matrix.shape)   # (n_features, latent_dim)
   print(model.lambda_matrix)        # Omega^T @ Omega

**WGRLVQ** — Adds per-feature relevance weighting:

.. math::

   W_2^2(x, k) = \sum_j \lambda_j (x_j - \mu_{kj})^2 + \sum_j \sigma_{kj}^2

where :math:`\lambda_j \geq 0` and :math:`\sum_j \lambda_j = 1`.

.. code-block:: python

   from prosemble.models import WGRLVQ

   model = WGRLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Learned relevance profile
   print(model.relevance_profile)   # shape: (n_features,), sums to 1

**WRSLVQ** — the probabilistic counterpart of WGLVQ.  The 2-Wasserstein
distance enters the robust soft LVQ cost with scores
:math:`f_k(x) = -W_2^2(x, k) / 2\sigma^2`.  Since
:math:`W_2^2(x, k) = \|x - \mu_k\|^2 + \operatorname{tr}\Sigma_k`, the
variance of a prototype acts as a learned prior weight
:math:`P(k) \propto \exp(-\operatorname{tr}\Sigma_k / 2\sigma^2)`:

.. code-block:: python

   from prosemble.models import WRSLVQ

   model = WRSLVQ(
       n_prototypes_per_class=2,
       sigma='auto',          # initial width from the data
       adapt_sigma=True,      # learn it during training
       max_iter=200,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   proba = model.predict_proba(X_test)
   labels = model.predict_with_rejection(X_test, confidence=0.8)

**WMRSLVQ** and **WLMRSLVQ** — WRSLVQ with an adaptive matrix.  The map
:math:`z \mapsto \Omega^T z` is applied to the sample and to the Gaussian
prototype alike, and the distance is the 2-Wasserstein distance between
the mapped measures:

.. math::

   W_2^2(x, k) = (x - \mu_k)^T \Lambda (x - \mu_k)
                 + \operatorname{tr}(\Lambda \Sigma_k),
   \qquad \Lambda = \Omega \Omega^T

This is the expected adaptive distance from the sample to a point drawn
from the prototype.  The variance of a feature counts in proportion to
the weight the metric gives that feature.  WMRSLVQ learns one matrix,
WLMRSLVQ one matrix :math:`\Lambda_k` per prototype:

.. code-block:: python

   from prosemble.models import WMRSLVQ, WLMRSLVQ

   model = WMRSLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01)
   model.fit(X_train, y_train)
   print(model.lambda_matrix.shape)    # (n_features, n_features)

   local = WLMRSLVQ(n_prototypes_per_class=2, latent_dim=None,
                    max_iter=200, lr=0.01)
   local.fit(X_train, y_train)
   print(local.lambda_matrix.shape)    # (n_prototypes, n_features, n_features)

.. note::

   Variances are parameterized internally as :math:`\log(\sigma^2)` to ensure
   positivity during gradient optimization.  Access the actual variances via
   ``model.prototype_variances_``.

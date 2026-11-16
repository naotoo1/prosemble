Concepts
========

.. currentmodule:: prosemble.models

Every model in Prosemble is built from the same few ideas. This page
introduces them; the :doc:`user guide </user_guide/index>` covers each
family in detail.

Prototypes
----------

A **prototype** is a point in the same space as the data, labelled with a
class (in a classifier) or standing for a cluster. A prototype classifier
assigns a sample the label of its nearest prototype. Each prototype
therefore owns a region of the input space, its *receptive field*, and the
decision boundary runs where receptive fields of different classes meet.

.. figure:: /_static/figures/hero-light.png
   :class: only-light pr-figure
   :width: 80%
   :align: center

.. figure:: /_static/figures/hero-dark.png
   :class: only-dark pr-figure
   :width: 80%
   :align: center

   Four prototypes per class (large markers) on the two-moons data. Shading
   shows the predicted class probability.

This makes the model easy to explain. A prediction says "this sample looks
most like that prototype", and the prototype can be shown, plotted or
compared with the sample feature by feature. A handful of prototypes per
class is usually enough.

Learning the prototypes
-----------------------

Generalized learning vector quantization (GLVQ) moves the prototypes by
gradient descent on the relative distance cost

.. math::

   \mu(x) = \frac{d^+(x) - d^-(x)}{d^+(x) + d^-(x)} \in [-1, 1],

where :math:`d^+` is the distance from a sample to the nearest prototype of
its own class and :math:`d^-` to the nearest prototype of another class.
A sample is classified correctly exactly when :math:`\mu < 0`, so
minimising :math:`\mu` pulls correct prototypes in and pushes wrong ones
away, and :math:`|\mu|` acts as a margin.

In Prosemble the cost is a JAX function of a dictionary of parameters
(prototypes, and any metric or width the model learns). Gradients come from
automatic differentiation, updates from an `Optax
<https://optax.readthedocs.io>`_ optimizer, and the whole training loop is
compiled into one XLA program with ``jax.lax.scan``.

The distance is the model
-------------------------

Most of the families differ only in how they measure distance. Because the
distance is learned, it is also an explanation of what the model learned.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Distance
     - What it learns
     - Models
   * - Squared Euclidean
     - Nothing; all features count equally.
     - :class:`GLVQ`
   * - Relevance-weighted
     - One weight per feature.
     - :class:`GRLVQ`
   * - Matrix (:math:`\|\Omega(x - w)\|^2`)
     - Weights of features and of pairs of features; a low-dimensional
       projection when :math:`\Omega` is rectangular.
     - :class:`GMLVQ`, :class:`LGMLVQ` (one matrix per prototype)
   * - Tangent
     - Directions of variation that should not count.
     - :class:`GTLVQ`
   * - Kernel
     - A non-linear similarity with learned parameters.
     - :doc:`Differentiating kernels </user_guide/kernels>`
   * - Geodesic
     - Distances on a curved space (rotations, covariance matrices, ...).
     - :doc:`Riemannian models </user_guide/riemannian>`
   * - 2-Wasserstein
     - A spread for every prototype.
     - :doc:`Distributional prototypes </user_guide/classification/wasserstein>`
   * - Soft-DTW divergence
     - Prototypes that are time series, compared after alignment.
     - :doc:`Time series </user_guide/time_series>`

Hard and soft models
--------------------

GLVQ-type models compare distances and give a label and a margin. **Soft**
(probabilistic) models instead treat each prototype as the centre of a
Gaussian component with width :math:`\sigma`. A class probability is the
share of the mixture density that belongs to the class, and training
maximises the likelihood of the correct class (robust soft LVQ,
:class:`RSLVQ`). Soft models give probabilities that can be calibrated and
used to reject uncertain samples; see :doc:`/user_guide/probabilities`.

Cooperation between prototypes
------------------------------

In plain LVQ only the nearest prototypes are updated for each sample. In
**Neural Gas** models every prototype is updated, weighted by its distance
rank, with a neighbourhood range :math:`\gamma` that shrinks during
training. This makes training robust to the initial prototypes. Self-
organizing maps (SOM) use a fixed grid instead of ranks, so neighbouring
grid nodes end up close in the data space, which makes them a tool for
visualization.

Beyond classification
---------------------

The same building blocks serve other tasks:

* **One-class classification** learns prototypes of a single, normal class
  and a threshold :math:`\theta_k` per prototype. A sample farther than the
  threshold from every prototype is an outlier. See
  :doc:`/user_guide/one_class`.
* **Clustering** learns prototypes (cluster centres) without labels. The
  fuzzy and possibilistic c-means models give each sample a degree of
  membership in every cluster. See :doc:`/user_guide/clustering`.
* **Topology learning** places prototypes to follow the shape of the data
  and connects neighbours: Neural Gas, growing Neural Gas and SOMs. See
  :doc:`/user_guide/topology`.

A common interface
------------------

All models follow the conventions of scikit-learn: hyperparameters go to the
constructor, ``fit`` learns from data, and fitted attributes end in an
underscore (``prototypes_``, ``loss_history_``). Supervised models have
``predict`` and ``predict_proba``; one-class models add
``decision_function``; clustering models have ``predict`` and memberships.
Models work with :class:`~prosemble.core.pipeline.Pipeline`,
:func:`~prosemble.core.model_selection.cross_val_score` and
:class:`~prosemble.core.model_selection.GridSearchCV`.

Glossary
--------

.. glossary::

   Prototype
      A learned point that stands for a class or a cluster, :math:`w_k`.

   Receptive field
      The region of the input space whose samples are nearest to a given
      prototype.

   Relevance
      A learned weight of a feature in the distance, :math:`\lambda_j`.

   Omega, Lambda
      The learned linear map :math:`\Omega` of matrix models and the
      matrix :math:`\Lambda = \Omega^\top \Omega` it induces; the diagonal
      of :math:`\Lambda` holds the feature relevances.

   Width
      The standard deviation :math:`\sigma` of the Gaussian components in a
      soft model.

   Neighbourhood range
      The parameter :math:`\gamma` (or :math:`\lambda`) that sets how many
      ranked prototypes share an update in Neural Gas models.

   Visibility threshold
      The radius :math:`\theta_k` around a prototype within which a one-class
      model accepts samples as normal.

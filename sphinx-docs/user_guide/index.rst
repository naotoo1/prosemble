User guide
==========

The user guide explains each model family and each part of the library:
what a model does, when to use it, and how to configure it. New to
prototype-based learning? Read :doc:`/getting_started/concepts` first.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`tag` Classification
      :link: classification/index
      :link-type: doc

      GLVQ and its metric-learning, probabilistic, Neural Gas,
      distributional and deep variants.

   .. grid-item-card:: :octicon:`shield-check` One-class classification
      :link: one_class
      :link-type: doc

      Learn what is normal from one class and flag everything else.

   .. grid-item-card:: :octicon:`circle` Clustering
      :link: clustering
      :link-type: doc

      Fuzzy, possibilistic and kernel c-means; k-means.

   .. grid-item-card:: :octicon:`share-android` Topology-preserving maps
      :link: topology
      :link-type: doc

      Neural Gas, growing Neural Gas and self-organizing maps.

   .. grid-item-card:: :octicon:`zap` Differentiating kernels
      :link: kernels
      :link-type: doc

      Kernel distances with learned parameters, across all tasks.

   .. grid-item-card:: :octicon:`pulse` Time series
      :link: time_series
      :link-type: doc

      Prototypes that are series, compared after a soft alignment.

   .. grid-item-card:: :octicon:`globe` Riemannian manifolds
      :link: riemannian
      :link-type: doc

      Prototypes on curved spaces: rotations, covariance matrices,
      hyperbolic space.

Working with models
-------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`graph` Probabilities and rejection
      :link: probabilities
      :link-type: doc

      Calibrate probabilities and abstain on uncertain samples.

   .. grid-item-card:: :octicon:`eye` Interpreting models
      :link: interpretation
      :link-type: doc

      Read prototypes, relevances and learned metrics.

   .. grid-item-card:: :octicon:`gear` Training
      :link: training
      :link-type: doc

      Mini-batches, early stopping, optimizers, schedules, regularization.

   .. grid-item-card:: :octicon:`beaker` Model selection
      :link: model_selection
      :link-type: doc

      Pipelines, cross-validation, grid search and statistical comparison.

   .. grid-item-card:: :octicon:`cpu` Performance and scaling
      :link: performance
      :link-type: doc

      GPUs, several devices, mixed precision.

   .. grid-item-card:: :octicon:`package` Saving and deployment
      :link: deployment
      :link-type: doc

      Save, quantize, export with JAX or to ONNX.

.. toctree::
   :hidden:
   :caption: Models

   classification/index
   one_class
   clustering
   topology
   time_series
   kernels
   riemannian

.. toctree::
   :hidden:
   :caption: Working with models

   probabilities
   interpretation
   training
   model_selection
   performance
   deployment

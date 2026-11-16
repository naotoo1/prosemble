Tutorials
=========

Complete worked examples. Each one trains a model on real or synthetic
data, shows what the model learned, and explains the result. The figures
are drawn by ``sphinx-docs/_scripts/make_figures.py``, which runs the same
code.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Interpretable classification
      :link: interpretable_classification
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-wine-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-wine-dark.png
         :class: only-dark

      Which measurements tell three wine cultivars apart, and a 2-D view of the data as the classifier sees it.

   .. grid-item-card:: Calibrated probabilities and rejection
      :link: probabilities
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-calibration-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-calibration-dark.png
         :class: only-dark

      Measure and fix an overconfident model, then set aside the samples it is unsure about.

   .. grid-item-card:: Anomaly detection
      :link: anomaly_detection
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/tutorial-anomaly-light.png
         :class: only-light

      .. image:: /_static/figures/tutorial-anomaly-dark.png
         :class: only-dark

      Learn the shape of normal data from normal samples alone and flag new outliers.

   .. grid-item-card:: Fuzzy clustering
      :link: clustering
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/tutorial-clustering-light.png
         :class: only-light

      .. image:: /_static/figures/tutorial-clustering-dark.png
         :class: only-dark

      Soft cluster memberships that reveal the samples between clusters.

   .. grid-item-card:: Learning the shape of data
      :link: topology
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-topology-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-topology-dark.png
         :class: only-dark

      Growing Neural Gas and self-organizing maps.

   .. grid-item-card:: Time-series classification
      :link: time_series
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-timeseries-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-timeseries-dark.png
         :class: only-dark

      Prototypes that are series: classify patterns that occur at different times.

   .. grid-item-card:: Classifying covariance matrices
      :link: covariance_matrices
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/tutorial-spd-curve-light.png
         :class: only-light

      .. image:: /_static/figures/tutorial-spd-curve-dark.png
         :class: only-dark

      Prototypes on the manifold of SPD matrices, against treating the matrices as vectors.

   .. grid-item-card:: Classifying orientations
      :link: rotations
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-rotations-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-rotations-dark.png
         :class: only-dark

      Rotations on SO(3), against quaternions and rotation matrices.

   .. grid-item-card:: Classifying subspaces
      :link: subspaces
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-subspaces-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-subspaces-dark.png
         :class: only-dark

      Sets of vectors as points of the Grassmannian, whatever their basis.

   .. grid-item-card:: Hierarchies in hyperbolic space
      :link: hyperbolic
      :link-type: doc
      :class-card: pr-gallery-card

      .. image:: /_static/figures/thumb-hyperbolic-light.png
         :class: only-light

      .. image:: /_static/figures/thumb-hyperbolic-dark.png
         :class: only-dark

      A tree in the Poincaré disk, where prototypes find the common ancestors.

.. toctree::
   :hidden:

   interpretable_classification
   probabilities
   anomaly_detection
   clustering
   topology
   time_series
   covariance_matrices
   rotations
   subspaces
   hyperbolic

More examples
-------------

The `examples directory <https://github.com/naotoo1/prosemble/tree/main/examples>`_
of the repository has a short script for most models, each on the Iris
data.

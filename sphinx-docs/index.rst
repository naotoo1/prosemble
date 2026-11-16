:html_theme.sidebar_secondary.remove: true

.. title:: Prosemble: interpretable prototype-based machine learning in JAX

.. raw:: html

   <section class="pr-hero">
     <h1>Prosemble</h1>
     <p class="pr-tagline">Interpretable prototype-based machine learning in JAX</p>
     <p class="pr-lead">
       Classifiers, anomaly detectors and clustering models whose decisions
       you can inspect: each one compares a sample with a few learned
       prototypes. Over 130 models, compiled with JAX for CPU, GPU and TPU.
     </p>
     <div class="pr-buttons">
       <a class="pr-primary" href="getting_started/index.html">Get started</a>
       <a class="pr-secondary" href="user_guide/index.html">User guide</a>
       <a class="pr-secondary" href="api/index.html">API reference</a>
     </div>
   </section>

.. grid:: 1 1 2 2
   :gutter: 4
   :class-container: pr-showcase

   .. grid-item::
      :columns: 12 12 6 6

      .. code-block:: bash

         pip install prosemble

      .. code-block:: python

         from prosemble.datasets import load_moons_jax
         from prosemble.models import GLVQ

         data = load_moons_jax(n_samples=300, noise=0.15)
         X, y = data.input_data, data.labels

         model = GLVQ(n_prototypes_per_class=4,
                      max_iter=300, lr=0.05)
         model.fit(X, y)

         model.predict(X)          # labels
         model.predict_proba(X)    # probabilities
         model.prototypes_         # the 8 prototypes

   .. grid-item::
      :columns: 12 12 6 6

      .. image:: _static/figures/hero-light.png
         :class: only-light pr-figure
         :alt: Decision regions of a GLVQ classifier with four prototypes per class on the two-moons data

      .. image:: _static/figures/hero-dark.png
         :class: only-dark pr-figure
         :alt: Decision regions of a GLVQ classifier with four prototypes per class on the two-moons data

.. rst-class:: pr-section-title

Why Prosemble
-------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: pr-feature

   .. grid-item-card:: :octicon:`eye` Interpretable by design

      A decision is a comparison with prototypes that live in the data
      space. Learned relevances and metrics show which features matter.

   .. grid-item-card:: :octicon:`stack` Over 130 models

      LVQ and metric learning, probabilistic and Neural Gas classifiers,
      one-class detectors, fuzzy clustering, SOMs, kernel, Riemannian,
      distributional, time-series and image models.

   .. grid-item-card:: :octicon:`zap` Fast with JAX

      Training loops compile to a single XLA program. Run on GPU or
      TPU, across several devices, in mixed precision.

   .. grid-item-card:: :octicon:`graph` Probabilities you can trust

      Soft models give mixture posteriors, calibrated by a learned
      temperature, and can reject samples they are unsure about.

   .. grid-item-card:: :octicon:`plug` A familiar interface

      ``fit``, ``predict`` and ``predict_proba``, with pipelines,
      cross-validation and grid search in the scikit-learn style.

   .. grid-item-card:: :octicon:`package` Ready to deploy

      Save and load, quantize, export a compiled JAX function, or export
      to ONNX to run without JAX.

.. rst-class:: pr-section-title

Explore the documentation
-------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: getting_started/index
      :link-type: doc

      Install Prosemble, train a first model, and learn the concepts.

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide/index
      :link-type: doc

      Every model family and feature, explained with examples.

   .. grid-item-card:: :octicon:`mortar-board` Tutorials
      :link: tutorials/index
      :link-type: doc

      Worked examples: interpretation, calibration, anomaly detection,
      clustering.

   .. grid-item-card:: :octicon:`meter` Benchmarks
      :link: benchmarks/index
      :link-type: doc

      Prototype models against standard classifiers, with statistical
      tests.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api/index
      :link-type: doc

      Every model, function and class, with its parameters.

   .. grid-item-card:: :octicon:`people` Development
      :link: development/index
      :link-type: doc

      Contribute, read the changelog, and cite Prosemble.

.. toctree::
   :hidden:

   Getting started <getting_started/index>
   User guide <user_guide/index>
   Tutorials <tutorials/index>
   Benchmarks <benchmarks/index>
   API reference <api/index>
   Development <development/index>

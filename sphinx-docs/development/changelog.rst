Changelog
=========

.. currentmodule:: prosemble.models

Notable changes in each release. Pull request numbers link to the
details on GitHub.

Unreleased
----------

**New models and features**

* :class:`SoftDTWMRSLVQ`: a soft-DTW prototype classifier for multivariate
  time series that learns a metric over the channels, with a benchmark on
  the UEA multivariate archive.
* Soft (probabilistic) variants of the kernel, Wasserstein and Riemannian
  families: :class:`DKRSLVQ`, :class:`DKMRSLVQ`, :class:`DKLMRSLVQ`,
  :class:`WRSLVQ`, :class:`WMRSLVQ`, :class:`WLMRSLVQ`,
  :class:`RiemannianRSLVQ`, :class:`RiemannianMRSLVQ`
  (`#110 <https://github.com/naotoo1/prosemble/pull/110>`_).
* One-class soft models trained by contrast with a reference distribution,
  which can be fitted to normal samples alone: :class:`OCDKRSLVQ`,
  :class:`OCWRSLVQ`, :class:`OCRiemannianRSLVQ`
  (`#111 <https://github.com/naotoo1/prosemble/pull/111>`_,
  `#112 <https://github.com/naotoo1/prosemble/pull/112>`_).
* Temperature calibration of the soft models, ``calibrate``
  (`#114 <https://github.com/naotoo1/prosemble/pull/114>`_).
* :mod:`prosemble.core.evaluation`: Friedman, Nemenyi and Holm-adjusted
  Wilcoxon tests, expected calibration error and selective accuracy; a
  36-dataset classification benchmark
  (`#115 <https://github.com/naotoo1/prosemble/pull/115>`_).
* :class:`SoftDTWRSLVQ`, a soft prototype classifier for time series under
  the soft-DTW divergence, with a benchmark on 24 UCR datasets
  (`#116 <https://github.com/naotoo1/prosemble/pull/116>`_).
* Intrinsic prototype updates for the supervised Riemannian models
  (`#106 <https://github.com/naotoo1/prosemble/pull/106>`_).
* The documentation was reorganised: getting-started pages, a user guide
  by task, tutorials, benchmarks, and one API page per model and function.

**Fixes**

* The contrastive one-class models (:class:`OCDKRSLVQ`, :class:`OCWRSLVQ`)
  scale the steps of their prototypes and kernel bandwidths by the spread
  of the target features, so that ``lr`` is relative to the data. On data
  with a small spread in many dimensions the first steps used to throw the
  prototypes beyond the targets, and :class:`OCDKRSLVQ` then gave every
  sample the same score.
* ``devices`` (multi-device training) and ``gradient_checkpointing`` are
  accepted by all supervised models; 76 did not accept them. Models that
  learn more than prototypes failed to predict after multi-device
  training (`#118 <https://github.com/naotoo1/prosemble/pull/118>`_).
* Four cases of silently wrong training results
  (`#104 <https://github.com/naotoo1/prosemble/pull/104>`_).
* Frozen parameters were updated by their raw gradient
  (`#107 <https://github.com/naotoo1/prosemble/pull/107>`_).
* Exact logarithm map for the tangent-space Riemannian models on SO(n)
  (`#108 <https://github.com/naotoo1/prosemble/pull/108>`_).
* A plain ``pip install`` is importable, and CI runs the full test suite
  (`#105 <https://github.com/naotoo1/prosemble/pull/105>`_).
* Dependency updates for security advisories
  (`#103 <https://github.com/naotoo1/prosemble/pull/103>`_).

2.4.0 (2026-08-30)
------------------

* Wasserstein LVQ (:class:`WGLVQ`, :class:`WGRLVQ`, :class:`WGMLVQ`) and the
  Poincaré-ball manifold for hyperbolic data
  (`#88 <https://github.com/naotoo1/prosemble/pull/88>`_).
* Separate learning rates for correct- and wrong-class prototypes in the
  supervised Neural Gas models (``lr_ratio``)
  (`#89 <https://github.com/naotoo1/prosemble/pull/89>`_,
  `#90 <https://github.com/naotoo1/prosemble/pull/90>`_).
* New models :class:`SCMNG`
  (`#90 <https://github.com/naotoo1/prosemble/pull/90>`_) and :class:`SNG`
  (`#93 <https://github.com/naotoo1/prosemble/pull/93>`_).
* One-class models: square-root reparameterisation of the thresholds,
  Voronoi-local initialisation, and formulations aligned with the original
  papers (`#98 <https://github.com/naotoo1/prosemble/pull/98>`_).

2.3.0 (2026-05-24)
------------------

* Twelve Riemannian differentiating-kernel models
  (`#85 <https://github.com/naotoo1/prosemble/pull/85>`_).
* Custom LVQ optimizers, the reject option, curriculum learning, prototype
  regularization and geodesic utilities
  (`#86 <https://github.com/naotoo1/prosemble/pull/86>`_).
* ONNX export with the reject option for supervised models
  (`#87 <https://github.com/naotoo1/prosemble/pull/87>`_).

2.2.0 (2026-05-23)
------------------

* Nine one-class and Neural Gas differentiating-kernel models, and ONNX
  export for all fifteen differentiating-kernel models
  (`#84 <https://github.com/naotoo1/prosemble/pull/84>`_).

2.1.2 (2026-05-23)
------------------

* Six differentiating-kernel models
  (`#83 <https://github.com/naotoo1/prosemble/pull/83>`_).

2.1.1 (2026-05-21)
------------------

* Multi-device data parallelism, gradient checkpointing and custom
  gradient rules (`#81 <https://github.com/naotoo1/prosemble/pull/81>`_).

2.1.0 (2026-05-16)
------------------

* Supervised Neural Gas models on Riemannian manifolds, and ONNX export,
  including Riemannian models on SO(n) and the Grassmannian
  (`#78 <https://github.com/naotoo1/prosemble/pull/78>`_,
  `#79 <https://github.com/naotoo1/prosemble/pull/79>`_).
* Complete parameter documentation and MathJax rendering in all model
  docstrings (`#68 <https://github.com/naotoo1/prosemble/pull/68>`_ to
  `#77 <https://github.com/naotoo1/prosemble/pull/77>`_).

2.0.0 (2026-05-10)
------------------

* :class:`SMNG`, :class:`SLNG` and :class:`STNG` registered as public
  models.

1.1.0 to 1.1.3 (2026-05-09 to 2026-05-10)
-----------------------------------------

* JAX-native :class:`~prosemble.core.pipeline.Pipeline` and
  :class:`~prosemble.core.model_selection.GridSearchCV`
  (`#58 <https://github.com/naotoo1/prosemble/pull/58>`_).
* One-class RSLVQ with Neural Gas cooperation
  (`#59 <https://github.com/naotoo1/prosemble/pull/59>`_).
* The cross-entropy Neural Gas family
  (`#60 <https://github.com/naotoo1/prosemble/pull/60>`_), matrix RSLVQ
  (:class:`MRSLVQ`, :class:`LMRSLVQ`,
  `#62 <https://github.com/naotoo1/prosemble/pull/62>`_) and the RSLVQ-NG
  family (`#64 <https://github.com/naotoo1/prosemble/pull/64>`_).

1.0.0 (2026-05-03)
------------------

* Complete migration to JAX, with 62 models
  (`#57 <https://github.com/naotoo1/prosemble/pull/57>`_).

Earlier releases
----------------

Releases 0.8.0 to 0.9.2 (2023–2024) predate the JAX implementation. See
the `GitHub releases <https://github.com/naotoo1/prosemble/releases>`_.

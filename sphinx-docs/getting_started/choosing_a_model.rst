Choosing a model
================

.. currentmodule:: prosemble.models

Prosemble has over 130 models, but most belong to a few families that combine
the same choices: the task, the distance, and whether the model is hard or
soft. Start from the task, then from what you need the model to tell you.

Classification
--------------

.. list-table::
   :header-rows: 1
   :widths: 38 27 35

   * - You want
     - Start with
     - Then try
   * - A simple, interpretable baseline
     - :class:`GLVQ`
     - :class:`CELVQ`, :class:`MedianLVQ` (prototypes are training samples)
   * - To know which features matter
     - :class:`GMLVQ`
     - :class:`GRLVQ` (one weight per feature), :class:`LGMLVQ` (local
       metrics)
   * - Probabilities you can trust, or to reject uncertain samples
     - :class:`DKLMRSLVQ`, calibrated
     - :class:`WLMRSLVQ`, :class:`RSLVQ`
   * - Classes made of several clusters, many prototypes per class
     - :class:`SRNG` or :class:`SMNG`
     - :class:`CELVQ_NG`, :class:`RSLVQ_NG`
   * - Non-linear boundaries without a neural network
     - :class:`DKGMLVQ`
     - The other :doc:`differentiating-kernel models </user_guide/kernels>`
   * - Data on a manifold (rotations, covariance matrices, hyperbolic
       embeddings)
     - :class:`RiemannianSRNG`
     - :class:`RiemannianRSLVQ`
   * - Time series
     - :class:`SoftDTWRSLVQ` with ``gamma=1``
     - Two prototypes per class
   * - Multivariate time series, and which channels matter
     - :class:`SoftDTWMRSLVQ`
     - :class:`SoftDTWLMRSLVQ` (one metric per prototype)
   * - Images
     - :class:`ImageGLVQ`
     - :class:`ImageGMLVQ`, :class:`ImageCBC`

On 36 tabular datasets (:doc:`/benchmarks/index`), :class:`DKLMRSLVQ` at
fixed settings was statistically level in accuracy with a random forest and
gradient boosting, and shared the lowest calibration error with a tuned RBF
SVM. The tuned SVM was more accurate.

One-class classification and anomaly detection
----------------------------------------------

When training data come from one class only and new samples must be
accepted or flagged:

* :class:`OCGLVQ` learns prototypes and a threshold for each; a good
  default.
* :class:`OCGMLVQ` adds a learned metric; :class:`OCRSLVQ` and the
  contrastive soft models (:class:`OCDKRSLVQ`, :class:`OCWRSLVQ`) give
  probabilities.
* :class:`SVQOCC` combines a Neural Gas ranking with response probabilities.

See :doc:`/user_guide/one_class`.

Clustering and topology
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 38 27 35

   * - You want
     - Start with
     - Then try
   * - Soft cluster memberships
     - :class:`FCM`
     - :class:`AFCM`, kernel variants such as :class:`KFCM`
   * - Clusters robust to noise and outliers
     - :class:`PFCM`
     - :class:`PCM`, :class:`IPCM`
   * - Hard clusters
     - :class:`KMeansPlusPlus`
     - :class:`HCM`
   * - A map of the data for visualization
     - :class:`KohonenSOM`
     - :class:`HeskesSOM`
   * - Prototypes that follow the data's shape
     - :class:`NeuralGas`
     - :class:`GrowingNeuralGas` (also learns how many prototypes)

Settings that matter
--------------------

* **Prototypes per class.** One or two are often enough; use more, with a
  Neural Gas model, when a class is spread over several regions.
* **Scale the features.** Distances depend on the units of the features;
  standardise them first, for example with
  :class:`~prosemble.core.pipeline.StandardScaler` in a
  :class:`~prosemble.core.pipeline.Pipeline`.
* **Iterations and learning rate.** ``max_iter`` and ``lr`` trade training
  time for fit; check ``loss_history_`` to see whether training has
  converged.
* **Tune on held-out data.** Use
  :class:`~prosemble.core.model_selection.GridSearchCV`; see
  :doc:`/user_guide/model_selection`.

ONNX Export
===========

Prosemble can export fitted models to `ONNX <https://onnx.ai/>`_ format for
cross-platform deployment.  An exported ONNX model reproduces the ``predict()``
output of the original model and runs anywhere ONNX Runtime is available —
no JAX or prosemble dependency needed at inference time.

**73 of 87 models** are supported.

Installation
------------

ONNX export requires the ``onnx`` package.  For inference, install
``onnxruntime`` as well:

.. code-block:: bash

   pip install onnx onnxruntime

Basic Usage
-----------

.. code-block:: python

   from prosemble.models import GLVQ
   from prosemble.core.onnx_export import export_onnx

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # Export to ONNX
   onnx_model = export_onnx(model, path='glvq_model.onnx')

The ``export_onnx`` function accepts:

- ``model`` — a fitted prosemble model
- ``batch_size`` — fixed batch dimension (default ``1``; use ``-1`` for
  dynamic batch size)
- ``opset_version`` — ONNX opset version (default ``17``)
- ``path`` — optional file path to save the ONNX model

Running with ONNX Runtime
--------------------------

.. code-block:: python

   import numpy as np
   import onnxruntime as ort

   session = ort.InferenceSession('glvq_model.onnx')
   X_test_np = np.asarray(X_test, dtype=np.float32)

   onnx_preds = session.run(None, {'X': X_test_np})[0]

The ONNX model takes a single input ``X`` of shape ``(batch_size, n_features)``
and returns an integer array of predicted labels.

Full Workflow Example
---------------------

.. code-block:: python

   import numpy as np
   from prosemble.models import GMLVQ
   from prosemble.datasets import load_iris_jax
   from prosemble.core.onnx_export import export_onnx
   import onnxruntime as ort

   # Train
   dataset = load_iris_jax()
   X, y = dataset.input_data, dataset.target_data
   model = GMLVQ(
       n_prototypes_per_class=1, max_iter=100,
       lr=0.01, latent_dim=2,
   )
   model.fit(X, y)

   # Export
   onnx_model = export_onnx(model, batch_size=-1, path='gmlvq.onnx')

   # Run with ONNX Runtime
   session = ort.InferenceSession('gmlvq.onnx')
   X_np = np.asarray(X, dtype=np.float32)
   onnx_preds = session.run(None, {'X': X_np})[0]

   # Verify
   jax_preds = model.predict(X)
   assert np.array_equal(jax_preds, onnx_preds)

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Family
     - Count
     - Models
   * - Supervised LVQ (squared Euclidean)
     - 9
     - GLVQ, GLVQ1, GLVQ21, LVQ1, LVQ21, MedianLVQ, CELVQ, SLVQ, RSLVQ
   * - Supervised LVQ (global omega)
     - 2
     - GMLVQ, MRSLVQ
   * - Supervised LVQ (local omega)
     - 2
     - LGMLVQ, LMRSLVQ
   * - Supervised LVQ (relevance-weighted)
     - 1
     - GRLVQ
   * - Supervised LVQ (tangent)
     - 1
     - GTLVQ
   * - Supervised NG (squared Euclidean)
     - 3
     - SRNG, CELVQ_NG, RSLVQ_NG
   * - Supervised NG (global omega)
     - 3
     - SMNG, MCELVQ_NG, MRSLVQ_NG
   * - Supervised NG (local omega)
     - 3
     - SLNG, LCELVQ_NG, LMRSLVQ_NG
   * - Supervised NG (tangent)
     - 2
     - STNG, TCELVQ_NG
   * - Unsupervised
     - 4
     - NeuralGas, GrowingNeuralGas, KohonenSOM, HeskesSOM
   * - Fuzzy clustering
     - 8
     - FCM, PCM, FPCM, PFCM, AFCM, HCM, IPCM, IPCM2
   * - One-class GLVQ
     - 10
     - OCGLVQ, OCGLVQ_NG, OCGRLVQ, OCGRLVQ_NG, OCGMLVQ, OCGMLVQ_NG, OCLGMLVQ, OCLGMLVQ_NG, OCGTLVQ, OCGTLVQ_NG
   * - One-class RSLVQ
     - 6
     - OCRSLVQ, OCRSLVQ_NG, OCMRSLVQ, OCMRSLVQ_NG, OCLMRSLVQ, OCLMRSLVQ_NG
   * - SVQ-OCC
     - 5
     - SVQOCC, SVQOCC_R, SVQOCC_M, SVQOCC_LM, SVQOCC_T
   * - MLP encoder + WTAC
     - 4
     - SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ, LVQMLN
   * - CNN encoder + WTAC
     - 3
     - ImageGLVQ, ImageGMLVQ, ImageGTLVQ
   * - PLVQ (Gaussian mixture)
     - 1
     - PLVQ
   * - CBC (reasoning matrices)
     - 2
     - CBC, ImageCBC
   * - Riemannian SO(n) (chordal)
     - 1
     - RiemannianSRNG
   * - Riemannian SO(n) (tangent-space metric)
     - 3
     - RiemannianSMNG, RiemannianSLNG, RiemannianSTNG
   * - Riemannian Grassmannian (tangent-space metric)
     - (same 3)
     - RiemannianSMNG, RiemannianSLNG, RiemannianSTNG (alternate manifold config)

Encoder Models
--------------

Models with MLP or CNN backbones (Siamese, Image, LVQMLN, PLVQ, ImageCBC)
are fully supported.  The ONNX graph encodes the backbone as standard ops:

- **MLP**: ``MatMul`` :math:`\rightarrow` ``Add`` :math:`\rightarrow`
  ``Activation`` per layer
- **CNN**: ``Conv`` (SAME padding) :math:`\rightarrow` ``Activation``
  per layer :math:`\rightarrow` ``GlobalAveragePool`` :math:`\rightarrow`
  ``Linear``

Prototypes are pre-computed in the latent space at export time, so only the
input needs to be encoded at runtime.  Supported activations: Sigmoid, ReLU,
Tanh, LeakyReLU, SELU.

Distance Functions in ONNX
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Distance
     - ONNX Implementation
     - Models
   * - Squared Euclidean
     - Expansion trick
     - GLVQ family, NG, NeuralGas, FCM, OC, SVQ-OCC, Encoders, CBC
   * - Global Omega
     - Project then squared Euclidean
     - GMLVQ, MRSLVQ, SMNG, OCGMLVQ, SVQOCC_M
   * - Local Omega
     - Batched MatMul per prototype
     - LGMLVQ, SLNG, OCLGMLVQ, SVQOCC_LM
   * - Tangent
     - Batched project-reconstruct
     - GTLVQ, STNG, OCGTLVQ, SVQOCC_T
   * - Relevance-weighted
     - Element-wise weighted squared diff
     - GRLVQ, OCGRLVQ, SVQOCC_R
   * - SO(n) Chordal
     - Broadcast subtract + Frobenius norm
     - RiemannianSRNG
   * - SO(n) Tangent
     - Skew-symmetric log map + metric adaptation
     - RiemannianSMNG/SLNG/STNG (SO)
   * - Grassmannian Tangent
     - Projection log map + metric adaptation
     - RiemannianSMNG/SLNG/STNG (Gr)

Not Supported
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Models
     - Count
     - Reason
   * - Kernel fuzzy clustering (KFCM, KPCM, KFPCM, KPFCM, KAFCM, KIPCM, KIPCM2)
     - 7
     - Gaussian kernel distance has no standard ONNX operator
   * - RiemannianNeuralGas
     - 1
     - Matrix logarithm via Schur decomposition has no ONNX operator
   * - Riemannian models + SPD(n) manifold
     - (config)
     - Eigendecomposition (eigh) has no ONNX operator
   * - RiemannianSRNG + Grassmannian manifold
     - (config)
     - SVD-based geodesic distance has no ONNX operator
   * - KNN
     - 1
     - k-nearest-neighbor logic, not prototype-based
   * - NPC
     - 1
     - Different predict pattern
   * - Utility (KMeansPlusPlus, Kmeans, SOM, BGPC)
     - 4
     - Not prototype model base classes

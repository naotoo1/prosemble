Models
======

.. currentmodule:: prosemble.models

All models are importable from :mod:`prosemble.models`. Each section links
to the user guide page that explains the family.

Classification
--------------

Prototype classifiers
~~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/classification/lvq`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   GLVQ
   GLVQ1
   GLVQ21
   CELVQ
   LVQ1
   LVQ21
   MedianLVQ

Metric learning
~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/classification/metric_learning`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   GRLVQ
   GMLVQ
   LGMLVQ
   GTLVQ

Probabilistic classifiers
~~~~~~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/classification/probabilistic`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SLVQ
   RSLVQ
   MRSLVQ
   LMRSLVQ

Neural Gas classifiers
~~~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/classification/neural_gas`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SNG
   SRNG
   SMNG
   SCMNG
   SLNG
   STNG
   CELVQ_NG
   MCELVQ_NG
   LCELVQ_NG
   TCELVQ_NG
   RSLVQ_NG
   MRSLVQ_NG
   LMRSLVQ_NG

Distributional prototypes
~~~~~~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/classification/wasserstein`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WGLVQ
   WGRLVQ
   WGMLVQ
   WRSLVQ
   WMRSLVQ
   WLMRSLVQ

Deep, Siamese and image models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/classification/deep`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LVQMLN
   PLVQ
   SiameseGLVQ
   SiameseGMLVQ
   SiameseGTLVQ
   ImageGLVQ
   ImageGMLVQ
   ImageGTLVQ
   CBC
   ImageCBC

Differentiating kernels
~~~~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/kernels`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   DKGLVQ
   DKGRLVQ
   DKGMLVQ
   DKRSLVQ
   DKMRSLVQ
   DKLMRSLVQ
   DKGLVQ_NG
   DKGRLVQ_NG
   DKGMLVQ_NG

Riemannian manifolds
~~~~~~~~~~~~~~~~~~~~

User guide: :doc:`/user_guide/riemannian`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RiemannianSRNG
   RiemannianSMNG
   RiemannianSLNG
   RiemannianSTNG
   RiemannianRSLVQ
   RiemannianMRSLVQ
   RiemannianDKGLVQ
   RiemannianDKGRLVQ
   RiemannianDKGMLVQ
   RiemannianDKSMNG
   RiemannianDKSLNG
   RiemannianDKSTNG
   RiemannianDKRSMNG
   RiemannianDKMSMNG
   RiemannianDKRSLNG
   RiemannianDKMSLNG
   RiemannianDKRSTNG
   RiemannianDKMSTNG

Time series
~~~~~~~~~~~

User guide: :doc:`/user_guide/time_series`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SoftDTWRSLVQ
   SoftDTWMRSLVQ

Nearest neighbours
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   KNN

One-class classification
------------------------

User guide: :doc:`/user_guide/one_class`

One-class GLVQ
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OCGLVQ
   OCGRLVQ
   OCGMLVQ
   OCLGMLVQ
   OCGTLVQ
   OCGLVQ_NG
   OCGRLVQ_NG
   OCGMLVQ_NG
   OCLGMLVQ_NG
   OCGTLVQ_NG

One-class RSLVQ
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OCRSLVQ
   OCMRSLVQ
   OCLMRSLVQ
   OCRSLVQ_NG
   OCMRSLVQ_NG
   OCLMRSLVQ_NG

SVQ-OCC
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SVQOCC
   SVQOCC_R
   SVQOCC_M
   SVQOCC_LM
   SVQOCC_T

Contrastive soft models
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OCDKRSLVQ
   OCWRSLVQ
   OCRiemannianRSLVQ

Differentiating kernels
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OCDKGLVQ
   OCDKGRLVQ
   OCDKGMLVQ
   OCDKGLVQ_NG
   OCDKGRLVQ_NG
   OCDKGMLVQ_NG

Clustering
----------

User guide: :doc:`/user_guide/clustering`

Fuzzy and possibilistic c-means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FCM
   PCM
   FPCM
   PFCM
   AFCM
   IPCM
   IPCM2
   NPC
   BGPC

Kernel c-means
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   KFCM
   KPCM
   KFPCM
   KPFCM
   KAFCM
   KIPCM
   KIPCM2

Hard clustering
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HCM
   Kmeans
   KMeansPlusPlus

Topology-preserving maps
------------------------

User guide: :doc:`/user_guide/topology`

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NeuralGas
   GrowingNeuralGas
   KohonenSOM
   HeskesSOM
   SOM
   RiemannianNeuralGas
   DKNeuralGas
   DKKohonenSOM
   DKHeskesSOM

Base classes
------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SupervisedPrototypeModel
   UnsupervisedPrototypeModel
   FuzzyClusteringBase

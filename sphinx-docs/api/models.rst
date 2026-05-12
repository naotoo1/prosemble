Models API Reference
====================

All models are importable from ``prosemble.models``.

Supervised LVQ
--------------

.. autoclass:: prosemble.models.GLVQ
   :members: fit, predict, predict_proba, save, load
   :undoc-members:

.. autoclass:: prosemble.models.GRLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.GMLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.LGMLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.GTLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.CELVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.LVQ1
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.LVQ21
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.MedianLVQ
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.SLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.RSLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.MRSLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.LMRSLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.RSLVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.MRSLVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.LMRSLVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.GLVQ1
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.GLVQ21
   :members: fit, predict
   :undoc-members:

Deep and Siamese Variants
--------------------------

.. autoclass:: prosemble.models.LVQMLN
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.PLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.SiameseGLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.SiameseGMLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.SiameseGTLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

Image LVQ
---------

.. autoclass:: prosemble.models.ImageGLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.ImageGMLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.ImageGTLVQ
   :members: fit, predict, predict_proba
   :undoc-members:

Classification-By-Components
-----------------------------

.. autoclass:: prosemble.models.CBC
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.ImageCBC
   :members: fit, predict, predict_proba
   :undoc-members:

Supervised Neural Gas
---------------------

.. autoclass:: prosemble.models.SRNG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.SMNG
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.SLNG
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.STNG
   :members: fit, predict
   :undoc-members:

Cross-Entropy Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prosemble.models.CELVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.MCELVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.LCELVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.TCELVQ_NG
   :members: fit, predict, predict_proba
   :undoc-members:

One-Class GLVQ
--------------

.. autoclass:: prosemble.models.OCGLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCGRLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCGMLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCLGMLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCGTLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

One-Class GLVQ with Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prosemble.models.OCGLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCGRLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCGMLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCLGMLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCGTLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

One-Class RSLVQ
---------------

.. autoclass:: prosemble.models.OCRSLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCMRSLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCLMRSLVQ
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

One-Class RSLVQ with Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prosemble.models.OCRSLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCMRSLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.OCLMRSLVQ_NG
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

SVQ-OCC
-------

.. autoclass:: prosemble.models.SVQOCC
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.SVQOCC_R
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.SVQOCC_M
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.SVQOCC_LM
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

.. autoclass:: prosemble.models.SVQOCC_T
   :members: fit, predict, decision_function, predict_with_reject
   :undoc-members:

Fuzzy Clustering
----------------

.. autoclass:: prosemble.models.FCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.PCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.FPCM
   :members: fit, predict, predict_proba, get_typicality
   :undoc-members:

.. autoclass:: prosemble.models.PFCM
   :members: fit, predict, predict_proba, predict_typicality
   :undoc-members:

.. autoclass:: prosemble.models.AFCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.HCM
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.IPCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.IPCM2
   :members: fit, predict, predict_proba
   :undoc-members:

Kernel Clustering
^^^^^^^^^^^^^^^^^

.. autoclass:: prosemble.models.KFCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.KPCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.KFPCM
   :members: fit, predict, predict_proba, get_typicality
   :undoc-members:

.. autoclass:: prosemble.models.KPFCM
   :members: fit, predict, predict_proba, get_typicality
   :undoc-members:

.. autoclass:: prosemble.models.KAFCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.KIPCM
   :members: fit, predict, predict_proba
   :undoc-members:

.. autoclass:: prosemble.models.KIPCM2
   :members: fit, predict, predict_proba
   :undoc-members:

Topology-Preserving Models
--------------------------

.. autoclass:: prosemble.models.NeuralGas
   :members: fit, predict, transform
   :undoc-members:

.. autoclass:: prosemble.models.GrowingNeuralGas
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.KohonenSOM
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.HeskesSOM
   :members: fit, predict
   :undoc-members:

Utility Models
--------------

.. autoclass:: prosemble.models.KMeansPlusPlus
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.KNN
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.NPC
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.Kmeans
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.SOM
   :members: fit, predict
   :undoc-members:

.. autoclass:: prosemble.models.BGPC
   :members: fit, predict
   :undoc-members:

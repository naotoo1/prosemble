"""
Prosemble models - JAX-based implementations
"""

# JAX-based models (require JAX installation)
try:
    from .base import FuzzyClusteringBase, NotFittedError
    from .prototype_base import SupervisedPrototypeModel, UnsupervisedPrototypeModel

    # Unsupervised clustering models
    from .afcm import AFCM
    from .bgpc import BGPC
    from .fcm import FCM
    from .fpcm import FPCM
    from .hcm import HCM
    from .ipcm import IPCM
    from .ipcm2 import IPCM2
    from .kafcm import KAFCM
    from .kfcm import KFCM
    from .kfpcm import KFPCM
    from .kipcm import KIPCM
    from .kipcm2 import KIPCM2
    from .kmeans import KMeansPlusPlus, Kmeans
    from .knn import KNN
    from .kpcm import KPCM
    from .kpfcm import KPFCM
    from .npc import NPC
    from .pcm import PCM
    from .pfcm import PFCM
    from .som import SOM

    # Supervised prototype models (LVQ family)
    from .glvq import GLVQ, GLVQ1, GLVQ21
    from .relevance_lvq import GRLVQ
    from .srng import SRNG
    from .smng import SMNG
    from .slng import SLNG
    from .stng import STNG
    from .matrix_lvq import GMLVQ
    from .local_matrix_lvq import LGMLVQ
    from .tangent_lvq import GTLVQ
    from .crossentropy_lvq import CELVQ
    from .celvq_ng import CELVQ_NG
    from .mcelvq_ng import MCELVQ_NG
    from .lcelvq_ng import LCELVQ_NG
    from .tcelvq_ng import TCELVQ_NG
    from .lvq1 import LVQ1
    from .lvq21 import LVQ21
    from .median_lvq import MedianLVQ
    from .probabilistic_lvq import SLVQ, RSLVQ
    from .matrix_rslvq import MRSLVQ, LMRSLVQ
    from .rslvq_ng import RSLVQ_NG
    from .mrslvq_ng import MRSLVQ_NG, LMRSLVQ_NG
    from .cbc import CBC
    from .lvqmln import LVQMLN
    from .plvq import PLVQ
    from .siamese_lvq import SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ
    from .image_lvq import ImageGLVQ, ImageGMLVQ, ImageGTLVQ
    from .image_cbc import ImageCBC

    # One-class models
    from .oc_glvq import OCGLVQ
    from .oc_grlvq import OCGRLVQ
    from .oc_gmlvq import OCGMLVQ
    from .oc_lgmlvq import OCLGMLVQ
    from .oc_gtlvq import OCGTLVQ
    from .oc_glvq_ng import OCGLVQ_NG
    from .oc_grlvq_ng import OCGRLVQ_NG
    from .oc_gmlvq_ng import OCGMLVQ_NG
    from .oc_lgmlvq_ng import OCLGMLVQ_NG
    from .oc_gtlvq_ng import OCGTLVQ_NG
    from .oc_rslvq import OCRSLVQ
    from .oc_mrslvq import OCMRSLVQ, OCLMRSLVQ
    from .oc_rslvq_ng import OCRSLVQ_NG
    from .oc_mrslvq_ng import OCMRSLVQ_NG, OCLMRSLVQ_NG
    from .svq_occ import SVQOCC
    from .svq_occ_r import SVQOCC_R
    from .svq_occ_m import SVQOCC_M
    from .svq_occ_lm import SVQOCC_LM
    from .svq_occ_t import SVQOCC_T

    # Unsupervised topology models
    from .neural_gas import NeuralGas
    from .growing_neural_gas import GrowingNeuralGas
    from .kohonen_som import KohonenSOM
    from .heskes_som import HeskesSOM

    _MODEL_REGISTRY = {
        # Unsupervised clustering
        'FCM': FCM, 'PCM': PCM, 'PFCM': PFCM, 'AFCM': AFCM,
        'FPCM': FPCM, 'HCM': HCM, 'IPCM': IPCM, 'IPCM2': IPCM2,
        'KFCM': KFCM, 'KPCM': KPCM, 'KAFCM': KAFCM, 'KFPCM': KFPCM,
        'KPFCM': KPFCM, 'KIPCM': KIPCM, 'KIPCM2': KIPCM2,
        # Supervised LVQ
        'GLVQ': GLVQ, 'GLVQ1': GLVQ1, 'GLVQ21': GLVQ21,
        'GRLVQ': GRLVQ, 'SRNG': SRNG, 'SMNG': SMNG, 'SLNG': SLNG, 'STNG': STNG,
        'GMLVQ': GMLVQ, 'LGMLVQ': LGMLVQ,
        'GTLVQ': GTLVQ, 'CELVQ': CELVQ, 'CELVQ_NG': CELVQ_NG,
        'MCELVQ_NG': MCELVQ_NG, 'LCELVQ_NG': LCELVQ_NG, 'TCELVQ_NG': TCELVQ_NG,
        'LVQ1': LVQ1, 'LVQ21': LVQ21, 'MedianLVQ': MedianLVQ,
        'SLVQ': SLVQ, 'RSLVQ': RSLVQ,
        'MRSLVQ': MRSLVQ, 'LMRSLVQ': LMRSLVQ,
        'RSLVQ_NG': RSLVQ_NG, 'MRSLVQ_NG': MRSLVQ_NG, 'LMRSLVQ_NG': LMRSLVQ_NG,
        'CBC': CBC,
        'LVQMLN': LVQMLN, 'PLVQ': PLVQ,
        'SiameseGLVQ': SiameseGLVQ, 'SiameseGMLVQ': SiameseGMLVQ,
        'SiameseGTLVQ': SiameseGTLVQ,
        'ImageGLVQ': ImageGLVQ, 'ImageGMLVQ': ImageGMLVQ,
        'ImageGTLVQ': ImageGTLVQ, 'ImageCBC': ImageCBC,
        # One-class GLVQ
        'OCGLVQ': OCGLVQ, 'OCGRLVQ': OCGRLVQ, 'OCGMLVQ': OCGMLVQ,
        'OCLGMLVQ': OCLGMLVQ, 'OCGTLVQ': OCGTLVQ,
        'OCGLVQ_NG': OCGLVQ_NG, 'OCGRLVQ_NG': OCGRLVQ_NG,
        'OCGMLVQ_NG': OCGMLVQ_NG, 'OCLGMLVQ_NG': OCLGMLVQ_NG,
        'OCGTLVQ_NG': OCGTLVQ_NG,
        # One-class RSLVQ
        'OCRSLVQ': OCRSLVQ, 'OCMRSLVQ': OCMRSLVQ, 'OCLMRSLVQ': OCLMRSLVQ,
        'OCRSLVQ_NG': OCRSLVQ_NG, 'OCMRSLVQ_NG': OCMRSLVQ_NG,
        'OCLMRSLVQ_NG': OCLMRSLVQ_NG,
        # One-class SVQ-OCC
        'SVQOCC': SVQOCC, 'SVQOCC_R': SVQOCC_R, 'SVQOCC_M': SVQOCC_M,
        'SVQOCC_LM': SVQOCC_LM, 'SVQOCC_T': SVQOCC_T,
        # Unsupervised topology
        'NeuralGas': NeuralGas, 'GrowingNeuralGas': GrowingNeuralGas,
        'KohonenSOM': KohonenSOM, 'HeskesSOM': HeskesSOM,
    }

    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    import warnings
    warnings.warn(f"JAX models not available. Install JAX to use prosemble models: pip install jax")

__all__ = [
    # Base classes
    'FuzzyClusteringBase',
    'SupervisedPrototypeModel',
    'UnsupervisedPrototypeModel',
    'NotFittedError',
    # Unsupervised clustering models
    'AFCM', 'BGPC', 'FCM', 'FPCM', 'HCM',
    'IPCM', 'IPCM2',
    'KAFCM', 'KFCM', 'KFPCM', 'KIPCM', 'KIPCM2',
    'KMeansPlusPlus', 'Kmeans',
    'KNN', 'KPCM', 'KPFCM', 'NPC',
    'PCM', 'PFCM', 'SOM',
    # Supervised LVQ family
    'GLVQ', 'GLVQ1', 'GLVQ21',
    'GRLVQ', 'SRNG', 'SMNG', 'SLNG', 'STNG',
    'GMLVQ', 'LGMLVQ', 'GTLVQ',
    'CELVQ', 'CELVQ_NG', 'MCELVQ_NG', 'LCELVQ_NG', 'TCELVQ_NG',
    'LVQ1', 'LVQ21', 'MedianLVQ',
    'SLVQ', 'RSLVQ', 'MRSLVQ', 'LMRSLVQ',
    'RSLVQ_NG', 'MRSLVQ_NG', 'LMRSLVQ_NG',
    'CBC',
    'LVQMLN', 'PLVQ',
    'SiameseGLVQ', 'SiameseGMLVQ', 'SiameseGTLVQ',
    'ImageGLVQ', 'ImageGMLVQ', 'ImageGTLVQ', 'ImageCBC',
    # One-class GLVQ
    'OCGLVQ', 'OCGRLVQ', 'OCGMLVQ', 'OCLGMLVQ', 'OCGTLVQ',
    'OCGLVQ_NG', 'OCGRLVQ_NG', 'OCGMLVQ_NG', 'OCLGMLVQ_NG', 'OCGTLVQ_NG',
    # One-class RSLVQ
    'OCRSLVQ', 'OCMRSLVQ', 'OCLMRSLVQ',
    'OCRSLVQ_NG', 'OCMRSLVQ_NG', 'OCLMRSLVQ_NG',
    # One-class SVQ-OCC
    'SVQOCC', 'SVQOCC_R', 'SVQOCC_M', 'SVQOCC_LM', 'SVQOCC_T',
    # Unsupervised topology
    'NeuralGas', 'GrowingNeuralGas', 'KohonenSOM', 'HeskesSOM',
    # Status
    'JAX_AVAILABLE',
]

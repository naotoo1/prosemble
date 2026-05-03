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
    from .matrix_lvq import GMLVQ
    from .local_matrix_lvq import LGMLVQ
    from .tangent_lvq import GTLVQ
    from .crossentropy_lvq import CELVQ
    from .lvq1 import LVQ1
    from .lvq21 import LVQ21
    from .median_lvq import MedianLVQ
    from .probabilistic_lvq import SLVQ, RSLVQ
    from .cbc import CBC
    from .lvqmln import LVQMLN
    from .plvq import PLVQ
    from .siamese_lvq import SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ
    from .image_lvq import ImageGLVQ, ImageGMLVQ, ImageGTLVQ
    from .image_cbc import ImageCBC

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
        'GRLVQ': GRLVQ, 'SRNG': SRNG,
        'GMLVQ': GMLVQ, 'LGMLVQ': LGMLVQ,
        'GTLVQ': GTLVQ, 'CELVQ': CELVQ,
        'LVQ1': LVQ1, 'LVQ21': LVQ21, 'MedianLVQ': MedianLVQ,
        'SLVQ': SLVQ, 'RSLVQ': RSLVQ,
        'CBC': CBC,
        'LVQMLN': LVQMLN, 'PLVQ': PLVQ,
        'SiameseGLVQ': SiameseGLVQ, 'SiameseGMLVQ': SiameseGMLVQ,
        'SiameseGTLVQ': SiameseGTLVQ,
        'ImageGLVQ': ImageGLVQ, 'ImageGMLVQ': ImageGMLVQ,
        'ImageGTLVQ': ImageGTLVQ, 'ImageCBC': ImageCBC,
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
    'GRLVQ', 'SRNG',
    'GMLVQ', 'LGMLVQ', 'GTLVQ',
    'CELVQ',
    'LVQ1', 'LVQ21', 'MedianLVQ',
    'SLVQ', 'RSLVQ',
    'CBC',
    'LVQMLN', 'PLVQ',
    'SiameseGLVQ', 'SiameseGMLVQ', 'SiameseGTLVQ',
    'ImageGLVQ', 'ImageGMLVQ', 'ImageGTLVQ', 'ImageCBC',
    # Unsupervised topology
    'NeuralGas', 'GrowingNeuralGas', 'KohonenSOM', 'HeskesSOM',
    # Status
    'JAX_AVAILABLE',
]

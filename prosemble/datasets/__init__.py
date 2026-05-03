"""
Prosemble datasets - JAX-based implementations
"""

from .dataset import *

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    import warnings
    warnings.warn("JAX is required for prosemble datasets. Install with: pip install jax")

__all__ = [
    'DATA',
    'DATA_JAX',
    'DATASET_JAX',
    'JAX_AVAILABLE',
]

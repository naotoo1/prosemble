"""
JAX-based clustering and classification models.

This package provides GPU-accelerated implementations of prototype-based
machine learning algorithms using JAX.
"""

from .fcm_jax import FCM_JAX
from .pcm_jax import PCM_JAX

__all__ = [
    'FCM_JAX',
    'PCM_JAX',
]

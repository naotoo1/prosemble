"""
JAX-based clustering and classification models.

This package provides GPU-accelerated implementations of prototype-based
machine learning algorithms using JAX.
"""

from .fcm_jax import FCM_JAX

__all__ = [
    'FCM_JAX',
]

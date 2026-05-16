"""Distributed training utilities for multi-device data parallelism.

Provides functions for sharding data across devices and replicating
model parameters, enabling data-parallel training on multi-GPU/TPU setups.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_mesh(devices=None):
    """Create a 1D device mesh for data parallelism.

    Parameters
    ----------
    devices : list of jax.Device or None
        Devices to use. If None, returns None (single-device mode).

    Returns
    -------
    Mesh or None
    """
    if devices is None:
        return None
    return Mesh(devices, axis_names=('data',))


def shard_data(X, y, mesh):
    """Shard data arrays along the batch dimension across devices.

    Parameters
    ----------
    X : jnp.ndarray of shape (n_samples, ...)
        Input data.
    y : jnp.ndarray of shape (n_samples,)
        Labels.
    mesh : Mesh
        Device mesh from create_mesh().

    Returns
    -------
    X_sharded, y_sharded : tuple of sharded arrays
    """
    data_sharding = NamedSharding(mesh, P('data'))
    X_sharded = jax.device_put(X, data_sharding)
    y_sharded = jax.device_put(y, data_sharding)
    return X_sharded, y_sharded


def replicate_params(params, mesh):
    """Replicate params across all devices (no partitioning).

    Parameters
    ----------
    params : dict (pytree)
        Model parameters.
    mesh : Mesh
        Device mesh from create_mesh().

    Returns
    -------
    params replicated across mesh
    """
    replicated = NamedSharding(mesh, P())
    return jax.device_put(params, replicated)


def replicate_opt_state(opt_state, mesh):
    """Replicate optimizer state across all devices.

    Parameters
    ----------
    opt_state : pytree
        Optax optimizer state.
    mesh : Mesh
        Device mesh from create_mesh().

    Returns
    -------
    opt_state replicated across mesh
    """
    replicated = NamedSharding(mesh, P())
    return jax.device_put(opt_state, replicated)


def unshard_params(params):
    """Bring params back to a single device.

    Used after training to store results as plain arrays for
    predict/export operations.

    Parameters
    ----------
    params : pytree
        Potentially sharded parameters.

    Returns
    -------
    params on default device
    """
    return jax.tree.map(
        lambda x: jax.device_put(x, jax.devices()[0]), params
    )

"""
Transfer/activation functions for prototype-based learning.

These functions are used to shape the GLVQ loss (mu values)
before summation, controlling the optimization landscape.
"""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def identity(x, beta=0.0):
    """Identity activation (passthrough).

    Parameters
    ----------
    x : array
        Input values.
    beta : float
        Ignored. Present for API consistency.

    Returns
    -------
    array
        Same as input.
    """
    return x


@jit
def sigmoid_beta(x, beta=10.0):
    """Sigmoid activation with steepness parameter.

    f(x) = 1 / (1 + exp(-beta * x))

    Parameters
    ----------
    x : array
        Input values.
    beta : float
        Steepness parameter. Higher values give sharper transition.

    Returns
    -------
    array
        Sigmoid-transformed values in (0, 1).
    """
    return jax.nn.sigmoid(beta * x)


@jit
def swish_beta(x, beta=10.0):
    """Swish activation with steepness parameter.

    f(x) = x * sigmoid(beta * x)

    Parameters
    ----------
    x : array
        Input values.
    beta : float
        Steepness parameter.

    Returns
    -------
    array
        Swish-transformed values.
    """
    return x * jax.nn.sigmoid(beta * x)


# Registry for name-based lookup
ACTIVATIONS = {
    'identity': identity,
    'sigmoid_beta': sigmoid_beta,
    'swish_beta': swish_beta,
}


def get_activation(name):
    """Get activation function by name.

    Parameters
    ----------
    name : str or callable
        Name of activation ('identity', 'sigmoid_beta', 'swish_beta')
        or a callable.

    Returns
    -------
    callable
        The activation function.
    """
    if callable(name):
        return name
    if name in ACTIVATIONS:
        return ACTIVATIONS[name]
    raise ValueError(
        f"Unknown activation '{name}'. "
        f"Available: {list(ACTIVATIONS.keys())}"
    )

"""
Structural typing protocols for prosemble interfaces.

Defines ``typing.Protocol`` contracts for duck-typed interfaces used
across the library.  These enable static type checking (mypy / pyright)
and IDE auto-completion without requiring inheritance.

.. note::

   Runtime-checkable protocols (``Manifold``, ``CallbackLike``) support
   ``isinstance()`` checks.  Type aliases (``DistanceMatrixFn``, etc.)
   are for annotation only.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Protocol,
    Tuple,
    runtime_checkable,
)

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Manifold protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Manifold(Protocol):
    """Protocol for Riemannian manifold implementations.

    Any object exposing the methods below can be used wherever a manifold
    is expected (e.g. ``RiemannianNeuralGas``, ``RiemannianSRNG``).

    The concrete implementations :class:`~prosemble.core.manifolds.SO`,
    :class:`~prosemble.core.manifolds.SPD`, and
    :class:`~prosemble.core.manifolds.Grassmannian` all satisfy this
    protocol structurally — no explicit subclassing is required.
    """

    @property
    def point_shape(self) -> Tuple[int, ...]:
        """Shape of a single point on the manifold."""
        ...

    def distance(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """Geodesic distance between two points.

        Parameters
        ----------
        p, q : arrays of shape ``point_shape``

        Returns
        -------
        scalar
        """
        ...

    def distance_squared(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """Squared geodesic distance between two points.

        Parameters
        ----------
        p, q : arrays of shape ``point_shape``

        Returns
        -------
        scalar
        """
        ...

    def log_map(self, base: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Logarithmic map: tangent vector at *base* pointing toward *target*.

        Parameters
        ----------
        base : array of shape ``point_shape``
            Base point on the manifold.
        target : array of shape ``point_shape``
            Target point on the manifold.

        Returns
        -------
        tangent : array of shape ``point_shape``
            Tangent vector in :math:`T_{\\text{base}} M`.
        """
        ...

    def exp_map(self, base: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
        """Exponential map: move along *tangent* from *base* back to the manifold.

        Parameters
        ----------
        base : array of shape ``point_shape``
        tangent : array of shape ``point_shape``

        Returns
        -------
        point : array of shape ``point_shape``
        """
        ...

    def random_point(self, key: jax.Array) -> jnp.ndarray:
        """Sample a random point on the manifold.

        Parameters
        ----------
        key : JAX PRNG key

        Returns
        -------
        point : array of shape ``point_shape``
        """
        ...

    def belongs(self, point: jnp.ndarray) -> jnp.ndarray:
        """Check whether *point* lies on the manifold.

        Parameters
        ----------
        point : array of shape ``point_shape``

        Returns
        -------
        bool or bool-valued array
        """
        ...

    def project(self, point: jnp.ndarray) -> jnp.ndarray:
        """Project an off-manifold point to the nearest point on the manifold.

        Parameters
        ----------
        point : array of shape ``point_shape``

        Returns
        -------
        projected : array of shape ``point_shape``
        """
        ...

    def injectivity_radius(self, point: jnp.ndarray) -> float:
        """Injectivity radius at *point*.

        The maximum geodesic distance for which the logarithmic map
        is injective.

        Parameters
        ----------
        point : array of shape ``point_shape``

        Returns
        -------
        radius : float or scalar array
        """
        ...


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CallbackLike(Protocol):
    """Protocol for training callbacks.

    Any object with the three hook methods below can be passed in the
    ``callbacks`` list of a model's constructor.  The existing
    :class:`~prosemble.core.callbacks.Callback` base class already
    satisfies this protocol.
    """

    def on_fit_start(self, model: Any, X: jnp.ndarray) -> None:
        """Called once before training begins."""
        ...

    def on_iteration_end(self, model: Any, info: dict) -> None:
        """Called after each training iteration / epoch."""
        ...

    def on_fit_end(self, model: Any, info: dict) -> None:
        """Called once after training ends."""
        ...


# ---------------------------------------------------------------------------
# Type aliases for callable interfaces
# ---------------------------------------------------------------------------

#: Distance-matrix function: ``(X, Y) -> distances``.
#: ``X`` has shape ``(n_samples, n_features)``,
#: ``Y`` has shape ``(n_prototypes, n_features)``,
#: result has shape ``(n_samples, n_prototypes)``.
DistanceMatrixFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

#: Pairwise distance function: ``(x, y) -> scalar``.
DistancePairwiseFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

#: Supervised prototype initializer:
#: ``(X, y, n_per_class, key) -> (prototypes, prototype_labels)``.
SupervisedInitFn = Callable[
    [jnp.ndarray, jnp.ndarray, int, jax.Array],
    Tuple[jnp.ndarray, jnp.ndarray],
]

#: Unsupervised prototype initializer:
#: ``(X, n_prototypes, key) -> prototypes``.
UnsupervisedInitFn = Callable[
    [jnp.ndarray, int, jax.Array],
    jnp.ndarray,
]

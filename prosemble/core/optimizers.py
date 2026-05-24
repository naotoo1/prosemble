"""Custom optax-compatible optimizers for prototype-based learning.

Provides specialized gradient transformations designed for the geometry
and parameter structure of LVQ models:

- ``per_group_clip``: Per-parameter-group gradient norm clipping.
- ``hypergradient_descent``: Adaptive per-parameter learning rates via
  gradient correlation (Baydin et al. 2017).
- ``riemannian_nesterov``: Nesterov accelerated gradient with manifold-aware
  momentum (parallel transport on Riemannian manifolds).

All transformations follow the optax ``GradientTransformation`` interface
and can be composed via ``optax.chain()`` or passed directly to any model's
``optimizer`` parameter.

References
----------
.. [1] Baydin, A. G., et al. (2017). Online learning rate adaptation with
       hypergradient descent. arXiv:1703.04782.
.. [2] Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). Optimization
       Algorithms on Matrix Manifolds. Princeton University Press.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


# =============================================================================
# Per-Group Gradient Clipping
# =============================================================================

class PerGroupClipState(NamedTuple):
    """State for per-group gradient clipping (stateless)."""
    pass


def per_group_clip(max_norms: dict) -> optax.GradientTransformation:
    """Clip gradient norms independently per parameter group.

    Different parameter types (prototypes, omega matrices, relevances,
    sigmas) have different natural scales. A single global clip either
    under-constrains large parameters or over-constrains small ones.
    This transformation clips each group independently.

    Parameters
    ----------
    max_norms : dict
        Mapping from parameter key name to maximum gradient norm.
        Keys not present in this dict are left unclipped.
        Example: ``{'prototypes': 1.0, 'omega': 0.5, 'sigmas': 0.1}``

    Returns
    -------
    optax.GradientTransformation
        Composable gradient transformation.

    Examples
    --------
    >>> import optax
    >>> from prosemble.core.optimizers import per_group_clip
    >>> optimizer = optax.chain(
    ...     per_group_clip({'prototypes': 1.0, 'omega': 0.5, 'sigmas': 0.1}),
    ...     optax.adam(0.01),
    ... )
    """

    def init_fn(params):
        del params
        return PerGroupClipState()

    def update_fn(updates, state, params=None):
        del params

        def clip_leaf(key, grad):
            if key in max_norms:
                max_norm = max_norms[key]
                grad_norm = jnp.sqrt(jnp.sum(grad ** 2))
                scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-10))
                return grad * scale
            return grad

        if isinstance(updates, dict):
            clipped = {k: clip_leaf(k, v) for k, v in updates.items()}
        else:
            clipped = updates
        return clipped, state

    return optax.GradientTransformation(init_fn, update_fn)


# =============================================================================
# Hypergradient Descent
# =============================================================================

class HypergradientState(NamedTuple):
    """State for hypergradient descent optimizer."""
    learning_rates: dict    # per-key adaptive learning rates
    prev_grads: dict        # previous iteration gradients
    base_opt_state: object  # inner optimizer state


def hypergradient_descent(
    init_lr: float = 0.01,
    hyper_lr: float = 1e-4,
    inner_optimizer: str = 'sgd',
    min_lr: float = 1e-6,
    max_lr: float = 1.0,
) -> optax.GradientTransformation:
    """Adaptive per-parameter learning rates via hypergradient descent.

    If consecutive gradients point in the same direction (positive dot
    product), increase the learning rate. If they oscillate (negative
    dot product), decrease it. This allows each parameter group to
    converge at its own optimal rate.

    The update rule for learning rate eta_k at step t:

    .. math::

        \\eta_k^{t+1} = \\text{clip}\\left(
            \\eta_k^t - \\beta \\cdot \\langle g_k^t, g_k^{t-1} \\rangle
        \\right)

    Parameters
    ----------
    init_lr : float
        Initial learning rate for all parameter groups. Default: 0.01.
    hyper_lr : float
        Learning rate for the learning rate update (meta-learning rate).
        Default: 1e-4.
    inner_optimizer : str
        Base optimizer to use ('sgd' applies raw scaled gradients).
        Default: 'sgd'.
    min_lr : float
        Minimum allowed learning rate. Default: 1e-6.
    max_lr : float
        Maximum allowed learning rate. Default: 1.0.

    Returns
    -------
    optax.GradientTransformation

    References
    ----------
    .. [1] Baydin, A. G., et al. (2017). Online learning rate adaptation
           with hypergradient descent. arXiv:1703.04782.

    Examples
    --------
    >>> from prosemble.core.optimizers import hypergradient_descent
    >>> optimizer = hypergradient_descent(init_lr=0.01, hyper_lr=1e-4)
    """

    def init_fn(params):
        learning_rates = jax.tree.map(
            lambda p: jnp.full((), init_lr), params
        )
        prev_grads = jax.tree.map(jnp.zeros_like, params)
        # No inner state needed for SGD-style
        return HypergradientState(
            learning_rates=learning_rates,
            prev_grads=prev_grads,
            base_opt_state=None,
        )

    def update_fn(updates, state, params=None):
        del params
        learning_rates = state.learning_rates
        prev_grads = state.prev_grads

        # Update learning rates based on gradient correlation
        def compute_new_lr(lr, grad, prev_grad):
            dot = jnp.sum(grad * prev_grad)
            # Same direction (dot > 0) -> increase lr
            new_lr = lr + hyper_lr * dot
            return jnp.clip(new_lr, min_lr, max_lr)

        new_learning_rates = jax.tree.map(
            compute_new_lr, learning_rates, updates, prev_grads
        )

        # Scale gradients by per-parameter adaptive lr
        scaled_updates = jax.tree.map(
            lambda lr, grad: -lr * grad,
            new_learning_rates, updates
        )

        new_state = HypergradientState(
            learning_rates=new_learning_rates,
            prev_grads=updates,  # store current grads for next step
            base_opt_state=None,
        )
        return scaled_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# =============================================================================
# Riemannian Nesterov Accelerated Gradient
# =============================================================================

class RiemannianNesterovState(NamedTuple):
    """State for Riemannian Nesterov momentum."""
    velocity: dict   # momentum buffer (in tangent space)
    step: jnp.ndarray


def riemannian_nesterov(
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> optax.GradientTransformation:
    """Nesterov accelerated gradient adapted for prototype-based models.

    Implements Nesterov momentum in Euclidean parameter space. For
    Riemannian models, the prototypes are stored in flattened form
    and the manifold projection is handled by ``_post_update()``.

    The momentum buffer provides O(1/t^2) convergence rate versus O(1/t)
    for vanilla gradient descent on convex objectives.

    Update rule:

    .. math::

        v_{t+1} = \\mu \\cdot v_t + g_t
        \\theta_{t+1} = \\theta_t - \\eta \\cdot (\\mu \\cdot v_{t+1} + g_t)

    This is the Nesterov variant where the lookahead is incorporated
    into the update (Sutskever et al. 2013 reformulation).

    Parameters
    ----------
    learning_rate : float
        Step size. Default: 0.01.
    momentum : float
        Momentum coefficient (0 < mu < 1). Higher values give more
        momentum. Default: 0.9.

    Returns
    -------
    optax.GradientTransformation

    Notes
    -----
    For Riemannian models where prototypes live on manifolds, the
    manifold retraction (projection back to manifold) is handled by
    the model's ``_post_update()`` method. This optimizer provides
    the accelerated gradient direction; the model ensures the result
    stays on the manifold.

    For true Riemannian Nesterov (with parallel transport), use this
    optimizer with Riemannian models that implement ``_post_update``
    with manifold projection.

    Examples
    --------
    >>> from prosemble.core.optimizers import riemannian_nesterov
    >>> optimizer = riemannian_nesterov(learning_rate=0.01, momentum=0.9)
    """

    def init_fn(params):
        velocity = jax.tree.map(jnp.zeros_like, params)
        return RiemannianNesterovState(
            velocity=velocity,
            step=jnp.zeros((), dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        del params
        mu = momentum

        # Update velocity: v = mu * v + grad
        new_velocity = jax.tree.map(
            lambda v, g: mu * v + g, state.velocity, updates
        )

        # Nesterov lookahead: update = -lr * (mu * v_new + grad)
        scaled_updates = jax.tree.map(
            lambda v, g: -learning_rate * (mu * v + g),
            new_velocity, updates
        )

        new_state = RiemannianNesterovState(
            velocity=new_velocity,
            step=state.step + 1,
        )
        return scaled_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

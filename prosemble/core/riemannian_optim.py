"""
Riemannian optimisation for manifold-valued prototypes.

A gradient-based optimiser moves a parameter by adding an update to it.
On a curved space this has two defects: the Euclidean gradient ignores
the metric, so the step length has no intrinsic meaning, and the sum
leaves the manifold and has to be projected back.

:func:`riemannian_update` removes both. It wraps any optax optimiser so
that, for the manifold-valued parameter,

1. the Euclidean gradient is replaced by the Riemannian gradient,
   expressed in an orthonormal frame of the tangent space, where the
   metric is the ordinary Euclidean one;
2. the wrapped optimiser computes its step in these coordinates;
3. the step is limited to a fraction ``tau`` of the injectivity radius,
   inside which the exponential map is a diffeomorphism;
4. the prototype is moved along the geodesic with the exponential map.

With plain SGD this is exactly Riemannian gradient descent,
:math:`w \\leftarrow \\operatorname{Exp}_w(-\\eta \\operatorname{grad} f(w))`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

#: Methods a manifold must provide for :func:`riemannian_update`.
REQUIRED_METHODS = (
    'gradient_to_frame', 'frame_to_tangent', 'tangent_norm',
    'exp_map', 'injectivity_radius',
)


def supports_riemannian_update(manifold) -> bool:
    """Whether *manifold* provides everything :func:`riemannian_update` needs."""
    return all(callable(getattr(manifold, m, None)) for m in REQUIRED_METHODS)


def _is_array(x) -> bool:
    return hasattr(x, 'shape') and hasattr(x, 'reshape')


def riemannian_update(optimizer, manifold, tau=0.95, param='prototypes'):
    """Make an optax optimiser act intrinsically on a manifold parameter.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        Optimiser whose step is to be taken in the tangent space.
    manifold : Manifold
        Must provide ``gradient_to_frame``, ``frame_to_tangent``,
        ``tangent_norm``, ``exp_map`` and ``injectivity_radius``.
    tau : float
        Safety factor in (0, 1]: the geodesic step length is limited to
        ``tau`` times the injectivity radius.
    param : str
        Key of the manifold-valued entry in the parameter dict. Its value
        has shape ``(n_points, prod(manifold.point_shape))``. All other
        entries are passed to ``optimizer`` unchanged.

    Returns
    -------
    optax.GradientTransformation
        The returned update for ``param`` equals
        :math:`\\operatorname{Exp}_w(v) - w`, so adding it to the
        parameter, as ``optax.apply_updates`` does, lands on the manifold.
    """
    point_shape = tuple(manifold.point_shape)

    def _to_frame(flat_w, flat_g):
        W = flat_w.reshape((-1,) + point_shape)
        G = flat_g.reshape(W.shape)
        return jax.vmap(manifold.gradient_to_frame)(W, G).reshape(flat_g.shape)

    def _geodesic_step(flat_w, flat_s):
        W = flat_w.reshape((-1,) + point_shape)
        S = flat_s.reshape(W.shape)

        def one(w, s):
            v = manifold.frame_to_tangent(w, s)
            norm = manifold.tangent_norm(w, v)
            max_norm = tau * manifold.injectivity_radius(w)
            scale = jnp.where(norm > max_norm, max_norm / (norm + 1e-12), 1.0)
            return manifold.exp_map(w, scale * v) - w

        return jax.vmap(one)(W, S).reshape(flat_s.shape)

    def init_fn(params):
        return optimizer.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "riemannian_update needs the current parameters: call "
                "update(grads, state, params)."
            )
        # The entry is absent or masked out when the prototypes are frozen.
        w = params.get(param) if isinstance(params, dict) else None
        g = updates.get(param) if isinstance(updates, dict) else None
        if not (_is_array(w) and _is_array(g)):
            return optimizer.update(updates, state, params)

        frame_grads = {**updates, param: _to_frame(w, g)}
        steps, new_state = optimizer.update(frame_grads, state, params)
        steps = {**steps, param: _geodesic_step(w, steps[param])}
        return steps, new_state

    return optax.GradientTransformation(init_fn, update_fn)

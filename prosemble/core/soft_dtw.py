"""
Soft dynamic time warping.

Dynamic time warping (DTW) compares two time series by the cheapest
monotone alignment of their time points. Soft-DTW replaces the minimum
over alignments by a soft minimum with smoothing :math:`\\gamma > 0`,

.. math::

    \\operatorname{sdtw}_\\gamma(x, y) = -\\gamma \\log \\sum_{A}
        \\exp\\left(-\\langle A, C(x, y) \\rangle / \\gamma\\right),

where :math:`A` runs over the alignment matrices and
:math:`C_{ij} = \\|x_i - y_j\\|^2`. It is differentiable in both series
and tends to DTW as :math:`\\gamma \\to 0`. Soft-DTW is not zero for
equal series and can be negative; the soft-DTW divergence

.. math::

    D_\\gamma(x, y) = \\operatorname{sdtw}_\\gamma(x, y)
        - \\tfrac12 \\operatorname{sdtw}_\\gamma(x, x)
        - \\tfrac12 \\operatorname{sdtw}_\\gamma(y, y)

is non-negative and zero only for equal series.

References
----------
.. [1] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss
       Function for Time-Series. Proceedings of the 34th International
       Conference on Machine Learning, PMLR 70, 894-903.
.. [2] Blondel, M., Mensch, A., & Vert, J.-P. (2021). Differentiable
       Divergences Between Time Series. Proceedings of the 24th
       International Conference on Artificial Intelligence and
       Statistics, PMLR 130, 3853-3861.
"""

from functools import partial

import jax
import jax.numpy as jnp

#: Stands in for an infinite accumulated cost. Finite, so that gradients
#: through cells outside the admissible region stay defined.
_BIG = 1e10


def _soft_min3(a, b, c, gamma):
    """:math:`-\\gamma \\log(e^{-a/\\gamma} + e^{-b/\\gamma} + e^{-c/\\gamma})`."""
    stacked = jnp.stack([a, b, c])
    return -gamma * jax.scipy.special.logsumexp(-stacked / gamma, axis=0)


def _cost_matrix(x, y):
    """Squared Euclidean ground cost between time points, shape (n, m)."""
    return jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)


def _accumulate(cost, gamma, band, soft):
    """Accumulated cost :math:`R_{n-1, m-1}` by anti-diagonal recursion.

    The recursion runs over the :math:`n + m - 1` anti-diagonals
    :math:`i + j = k`; each step updates a whole diagonal at once.
    """
    n, m = cost.shape
    rows = jnp.arange(n)

    def diagonal(k):
        cols = k - rows
        valid = (cols >= 0) & (cols < m)
        if band is not None:
            # Sakoe-Chiba band, scaled to the lengths of both series.
            valid = valid & (jnp.abs(rows / max(n - 1, 1) - cols / max(m - 1, 1)) <= band)
        values = cost[rows, jnp.clip(cols, 0, m - 1)]
        return jnp.where(valid, values, _BIG), valid

    def shift(d, first):
        """Value at row ``i - 1``, with ``first`` above the first row."""
        return jnp.concatenate([jnp.array([first], dtype=d.dtype), d[:-1]])

    def step(carry, k):
        before, last = carry                       # diagonals k - 2 and k - 1
        values, valid = diagonal(k)
        corner = jnp.where(k == 0, 0.0, _BIG)      # R_{-1,-1} = 0
        diag = shift(before, corner)               # R_{i-1, j-1}
        up = shift(last, _BIG)                     # R_{i-1, j}
        left = last                                # R_{i, j-1}
        if soft:
            best = _soft_min3(diag, up, left, gamma)
        else:
            best = jnp.minimum(jnp.minimum(diag, up), left)
        current = jnp.where(valid, values + best, _BIG)
        return (last, current), None

    start = jnp.full(n, _BIG, dtype=cost.dtype)
    (_, final), _ = jax.lax.scan(step, (start, start), jnp.arange(n + m - 1))
    return final[n - 1]


@partial(jax.jit, static_argnames=('band',))
def soft_dtw(x, y, gamma=1.0, band=None):
    """Soft-DTW between two time series.

    Parameters
    ----------
    x : array of shape (n,) or (n, channels)
    y : array of shape (m,) or (m, channels)
    gamma : float
        Smoothing; DTW is the limit ``gamma -> 0``.
    band : float, optional
        Sakoe-Chiba band as a fraction of the series length. Alignments
        that leave the band are excluded. Default: no band.

    Returns
    -------
    value : scalar
    """
    x = x[:, None] if x.ndim == 1 else x
    y = y[:, None] if y.ndim == 1 else y
    return _accumulate(_cost_matrix(x, y), gamma, band, soft=True)


@partial(jax.jit, static_argnames=('band',))
def dtw(x, y, band=None):
    """Dynamic time warping with the squared Euclidean ground cost.

    Parameters
    ----------
    x : array of shape (n,) or (n, channels)
    y : array of shape (m,) or (m, channels)
    band : float, optional
        Sakoe-Chiba band as a fraction of the series length.

    Returns
    -------
    value : scalar
        Cost of the cheapest alignment.
    """
    x = x[:, None] if x.ndim == 1 else x
    y = y[:, None] if y.ndim == 1 else y
    return _accumulate(_cost_matrix(x, y), 0.0, band, soft=False)


@partial(jax.jit, static_argnames=('band',))
def soft_dtw_divergence(x, y, gamma=1.0, band=None):
    """Soft-DTW divergence: non-negative, and zero only for equal series.

    Parameters
    ----------
    x : array of shape (n,) or (n, channels)
    y : array of shape (m,) or (m, channels)
    gamma : float
        Smoothing.
    band : float, optional
        Sakoe-Chiba band as a fraction of the series length.

    Returns
    -------
    value : scalar
    """
    return (soft_dtw(x, y, gamma, band)
            - 0.5 * soft_dtw(x, x, gamma, band)
            - 0.5 * soft_dtw(y, y, gamma, band))


@partial(jax.jit, static_argnames=('band',))
def soft_dtw_matrix(X, Y, gamma=1.0, band=None):
    """Soft-DTW between every series of ``X`` and every series of ``Y``.

    Parameters
    ----------
    X : array of shape (n_x, length) or (n_x, length, channels)
    Y : array of shape (n_y, length) or (n_y, length, channels)

    Returns
    -------
    values : array of shape (n_x, n_y)
    """
    pair = lambda x, y: soft_dtw(x, y, gamma, band)
    return jax.vmap(lambda x: jax.vmap(lambda y: pair(x, y))(Y))(X)


@partial(jax.jit, static_argnames=('band',))
def soft_dtw_divergence_matrix(X, Y, gamma=1.0, band=None):
    """Soft-DTW divergence between every series of ``X`` and of ``Y``.

    Parameters
    ----------
    X : array of shape (n_x, length) or (n_x, length, channels)
    Y : array of shape (n_y, length) or (n_y, length, channels)

    Returns
    -------
    values : array of shape (n_x, n_y)
    """
    self_x = jax.vmap(lambda x: soft_dtw(x, x, gamma, band))(X)
    self_y = jax.vmap(lambda y: soft_dtw(y, y, gamma, band))(Y)
    return soft_dtw_matrix(X, Y, gamma, band) - 0.5 * self_x[:, None] - 0.5 * self_y[None, :]


@partial(jax.jit, static_argnames=('band',))
def dtw_matrix(X, Y, band=None):
    """DTW between every series of ``X`` and every series of ``Y``.

    Parameters
    ----------
    X : array of shape (n_x, length) or (n_x, length, channels)
    Y : array of shape (n_y, length) or (n_y, length, channels)

    Returns
    -------
    values : array of shape (n_x, n_y)
    """
    return jax.vmap(lambda x: jax.vmap(lambda y: dtw(x, y, band))(Y))(X)

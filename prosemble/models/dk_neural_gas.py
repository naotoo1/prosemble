"""
Differentiating Kernel Neural Gas (DKNeuralGas).

Neural Gas with Gaussian kernel distance in feature space:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|x - w_k\\|^2}{2\\sigma^2}
    \\right)\\right)

The kernel bandwidth :math:`\\sigma` is a fixed hyperparameter.
All other NG mechanics (ranking, neighborhood, exponential decay)
remain unchanged. Prototypes live in the original data space.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from prosemble.models.neural_gas import NeuralGas, NGState
from prosemble.core.kernel import kernel_distance_squared_per_proto


class DKNeuralGas(NeuralGas):
    """Differentiating Kernel Neural Gas.

    Neural Gas with Gaussian kernel distance for ranking. The kernel
    bandwidth :math:`\\sigma` is a fixed hyperparameter (not learned).
    The competitive Hebbian update rule operates in the original data
    space — only the distance metric changes.

    Parameters
    ----------
    kernel_sigma : float
        Gaussian kernel bandwidth. Default: 1.0.
    n_prototypes : int
        Number of prototypes/nodes.
    lr_init : float
        Initial learning rate.
    lr_final : float
        Final learning rate.
    lambda_init : float, optional
        Initial neighborhood range.
    lambda_final : float
        Final neighborhood range.
    max_iter : int
        Maximum training iterations.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    callbacks : list, optional
        Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training.
    patience : int, optional
        Epochs with no improvement before early stopping.
    restore_best : bool
        If True, restore best parameters after training.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.

    See Also
    --------
    NeuralGas : Base class with Euclidean distance.
    """

    def __init__(self, n_prototypes, kernel_sigma=1.0, lr_init=0.5,
                 lr_final=0.01, lambda_init=None, lambda_final=0.01,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 distance_fn=None, callbacks=None, use_scan=True,
                 patience=None, restore_best=False):
        super().__init__(
            n_prototypes=n_prototypes, lr_init=lr_init, lr_final=lr_final,
            lambda_init=lambda_init, lambda_final=lambda_final,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
            callbacks=callbacks, use_scan=use_scan,
            patience=patience, restore_best=restore_best,
        )
        self.kernel_sigma = kernel_sigma

    def _kernel_distances(self, X, prototypes):
        """Compute kernel distances with broadcast sigma."""
        sigmas = jnp.full(prototypes.shape[0], self.kernel_sigma)
        return kernel_distance_squared_per_proto(X, prototypes, sigmas)

    @partial(jit, static_argnums=(0,))
    def _ng_step(self, state, X, lambda_init):
        """Single JIT-compiled DK Neural Gas training step."""
        t = state.iteration
        max_t = jnp.array(max(self.max_iter - 1, 1), dtype=jnp.float32)
        frac = t.astype(jnp.float32) / max_t

        lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac
        lam_t = lambda_init * (self.lambda_final / lambda_init) ** frac

        prototypes = state.prototypes
        distances = self._kernel_distances(X, prototypes)

        # Rank prototypes for each sample
        order = jnp.argsort(distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
        h = jnp.exp(-ranks / lam_t)

        # Weighted update in data space
        diffs = X[:, None, :] - prototypes[None, :, :]
        weighted_diffs = h[:, :, None] * diffs
        update = lr_t * jnp.mean(weighted_diffs, axis=0)

        new_prototypes = prototypes + update
        energy = jnp.sum(h * distances)

        # Convergence
        has_converged = state.converged | (
            jnp.abs(energy - state.prev_loss) < self.epsilon
        )
        frozen_prototypes = jnp.where(state.converged, prototypes, new_prototypes)
        frozen_energy = jnp.where(state.converged, state.loss, energy)

        new_state = NGState(
            prototypes=frozen_prototypes,
            loss=frozen_energy,
            prev_loss=energy,
            converged=has_converged,
            iteration=t + 1,
        )
        return new_state, frozen_energy

    def _fit_with_python_loop(self, X, prototypes, lambda_init_val):
        """Python for-loop training with kernel distance."""
        loss_history = []

        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac
            lam_t = lambda_init_val * (self.lambda_final / lambda_init_val) ** frac

            distances = self._kernel_distances(X, prototypes)
            order = jnp.argsort(distances, axis=1)
            ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
            h = jnp.exp(-ranks / lam_t)

            diffs = X[:, None, :] - prototypes[None, :, :]
            weighted_diffs = h[:, :, None] * diffs
            update = lr_t * jnp.mean(weighted_diffs, axis=0)
            prototypes = prototypes + update

            energy = float(jnp.sum(h * distances))
            loss_history.append(energy)

            if t > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        self.prototypes_ = prototypes
        self.n_iter_ = t + 1
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        return self

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['kernel_sigma'] = self.kernel_sigma
        return hp

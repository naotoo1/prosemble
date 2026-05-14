"""
Neural Gas algorithm.

Topology-preserving unsupervised learning with rank-based
neighborhood adaptation and exponential decay.

References
----------
.. [1] Martinetz, T. M., Berkovich, S. G., & Schulten, K. J. (1993).
       "Neural-gas" network for vector quantization and its application
       to time-series prediction. IEEE Trans. Neural Networks.
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from prosemble.models.prototype_base import UnsupervisedPrototypeModel
from prosemble.core.distance import squared_euclidean_distance_matrix


class NGState(NamedTuple):
    """State for Neural Gas lax.scan loop."""
    prototypes: jnp.ndarray
    loss: jnp.ndarray
    prev_loss: jnp.ndarray
    converged: jnp.ndarray
    iteration: jnp.ndarray


class NeuralGas(UnsupervisedPrototypeModel):
    """Neural Gas.

    Updates all prototypes based on rank-distance:
        h(rank, lambda) = exp(-rank / lambda)
        w_k += lr * h(rank_k) * (x - w_k)

    Both lr and lambda decay exponentially during training.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes.
    max_iter : int
        Maximum training epochs.
    lr_init : float
        Initial learning rate.
    lr_final : float
        Final learning rate.
    lambda_init : float
        Initial neighborhood range.
    lambda_final : float
        Final neighborhood range.

    See Also
    --------
    UnsupervisedPrototypeModel : Full list of base parameters (distance_fn,
        callbacks, use_scan, patience, etc.).
    """

    def __init__(self, n_prototypes, lr_init=0.5, lr_final=0.01,
                 lambda_init=None, lambda_final=0.01,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 distance_fn=None, callbacks=None, use_scan=True,
                 patience=None, restore_best=False):
        super().__init__(
            n_prototypes=n_prototypes, max_iter=max_iter, lr=lr,
            epsilon=epsilon, random_seed=random_seed, distance_fn=distance_fn,
            callbacks=callbacks, use_scan=use_scan, patience=patience,
            restore_best=restore_best,
        )
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lambda_init = lambda_init  # defaults to n_prototypes/2
        self.lambda_final = lambda_final

    @partial(jit, static_argnums=(0,))
    def _ng_step(self, state, X, lambda_init):
        """Single JIT-compiled Neural Gas training step."""
        t = state.iteration
        max_t = jnp.array(max(self.max_iter - 1, 1), dtype=jnp.float32)
        frac = t.astype(jnp.float32) / max_t

        lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac
        lam_t = lambda_init * (self.lambda_final / lambda_init) ** frac

        prototypes = state.prototypes
        distances = squared_euclidean_distance_matrix(X, prototypes)

        # Rank prototypes for each sample
        order = jnp.argsort(distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
        h = jnp.exp(-ranks / lam_t)

        # Weighted update
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

    @partial(jit, static_argnums=(0,))
    def _fit_scan(self, X, prototypes, lambda_init):
        """Scan-based training loop."""
        initial_state = NGState(
            prototypes=prototypes,
            loss=jnp.array(float('inf')),
            prev_loss=jnp.array(float('inf')),
            converged=jnp.array(False),
            iteration=jnp.array(0),
        )

        def scan_fn(state, _):
            return self._ng_step(state, X, lambda_init)

        final_state, loss_history = lax.scan(
            scan_fn, initial_state, None, length=self.max_iter
        )
        return final_state, loss_history

    def fit(self, X):
        """Fit Neural Gas."""
        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples = X.shape[0]

        key = self.key
        indices = jax.random.choice(key, n_samples, (self.n_prototypes,), replace=False)
        prototypes = X[indices]

        lambda_init_val = self.lambda_init if self.lambda_init else self.n_prototypes / 2.0

        if self.use_scan:
            return self._fit_with_scan(X, prototypes, lambda_init_val)
        else:
            return self._fit_with_python_loop(X, prototypes, lambda_init_val)

    def _fit_with_scan(self, X, prototypes, lambda_init_val):
        """lax.scan training: JIT-compiled, runs all max_iter iterations."""
        lambda_init = jnp.array(lambda_init_val, dtype=jnp.float32)
        final_state, loss_history = self._fit_scan(X, prototypes, lambda_init)

        converged_mask = jnp.abs(jnp.diff(loss_history)) < self.epsilon
        first_converged = jnp.argmax(converged_mask)
        has_any = jnp.any(converged_mask)
        n_iter = jnp.where(has_any, first_converged + 2, self.max_iter)

        self.prototypes_ = final_state.prototypes
        self.n_iter_ = int(n_iter)
        self.loss_ = float(final_state.loss)
        self.loss_history_ = loss_history
        return self

    def _fit_with_python_loop(self, X, prototypes, lambda_init_val):
        """Python for-loop training: true early stopping, no wasted compute."""
        loss_history = []

        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac
            lam_t = lambda_init_val * (self.lambda_final / lambda_init_val) ** frac

            distances = squared_euclidean_distance_matrix(X, prototypes)
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
        hp.update({
            'lr_init': self.lr_init,
            'lr_final': self.lr_final,
            'lambda_final': self.lambda_final,
        })
        if self.lambda_init is not None:
            hp['lambda_init'] = self.lambda_init
        return hp

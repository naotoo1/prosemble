"""
Riemannian Neural Gas (RNG).

Generalizes the Neural Gas algorithm to Riemannian manifolds by
replacing Euclidean distance with geodesic distance and using
exponential/logarithmic maps for prototype updates:

.. math::

    w_i^{\\text{new}} \\leftarrow \\text{Exp}_{w_i}\\bigl(\\varepsilon \\cdot h_\\lambda(k_i(x, W)) \\cdot \\text{Log}_{w_i}(x)\\bigr)

where :math:`\\text{Exp}` and :math:`\\text{Log}` are the Riemannian exponential and
logarithmic maps, and :math:`h_\\lambda(k) = \\exp(-k/\\lambda)` is the rank-based
neighborhood function.

The algorithm learns prototypes that lie on the manifold, respecting
its intrinsic geometry and curvature.

References
----------
.. [1] Schwarz, L., Psenickova, M., Villmann, T., & Rohrbein, F. (2026).
       Topology-Preserving Prototype Learning on Riemannian Manifolds.
       ESANN 2026.
.. [2] Martinetz, T., Berkovich, S., & Schulten, K. (1993).
       Neural-gas network for vector quantization. IEEE Trans. NN.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import UnsupervisedPrototypeModel


class RiemannianNeuralGas(UnsupervisedPrototypeModel):
    """Riemannian Neural Gas.

    Learns prototypes on a Riemannian manifold using geodesic distances
    and exponential/logarithmic maps for updates.

    Parameters
    ----------
    manifold : object
        Manifold instance (SO, SPD, or Grassmannian from
        ``prosemble.core.manifolds``). Must implement ``distance``,
        ``log_map``, ``exp_map``, ``random_point``, ``project``,
        and ``injectivity_radius``.
    n_prototypes : int
        Number of prototypes/nodes.
    lr_init : float
        Initial learning rate. Default: 0.3.
    lr_final : float
        Final learning rate. Default: 0.01.
    lambda_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    lambda_final : float
        Final neighborhood range. Default: 0.01.
    tau : float
        Safety factor for injectivity radius bound. Default: 0.9.
    max_iter : int
        Maximum training iterations.
    lr : float
        Initial learning rate.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    distance_fn : callable, optional
        Distance function.
    callbacks : list, optional
        Callback objects.
    patience : int, optional
        Epochs with no improvement before early stopping. Default: None.
    restore_best : bool
        If True, restore parameters from the lowest-loss epoch. Default: False.
    """

    def __init__(self, manifold, n_prototypes, lr_init=0.3, lr_final=0.01,
                 lambda_init=None, lambda_final=0.01,
                 tau=0.9, max_iter=100, lr=0.01, epsilon=1e-6,
                 random_seed=42, distance_fn=None, callbacks=None,
                 patience=None, restore_best=False):
        # RNG uses Python loop — funm may not be scan-compatible
        super().__init__(
            n_prototypes=n_prototypes, max_iter=max_iter, lr=lr,
            epsilon=epsilon, random_seed=random_seed, distance_fn=distance_fn,
            callbacks=callbacks, use_scan=False, patience=patience,
            restore_best=restore_best,
        )
        self.manifold = manifold
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.tau = tau

    def _compute_distance_matrix(self, X, prototypes):
        """Compute pairwise geodesic distance matrix.

        Parameters
        ----------
        X : array of shape (n_samples, *point_shape)
        prototypes : array of shape (n_prototypes, *point_shape)

        Returns
        -------
        distances : array of shape (n_samples, n_prototypes)
        """
        def dist_one(x, w):
            return self.manifold.distance_squared(x, w)

        # vmap over prototypes, then over samples
        dist_to_all = jax.vmap(dist_one, in_axes=(None, 0))
        dist_matrix = jax.vmap(dist_to_all, in_axes=(0, None))
        return dist_matrix(X, prototypes)

    def _compute_updates(self, X, prototypes, h):
        """Compute rank-weighted tangent vectors for each prototype.

        Parameters
        ----------
        X : array of shape (n_samples, *point_shape)
        prototypes : array of shape (n_prototypes, *point_shape)
        h : array of shape (n_samples, n_prototypes) — cooperation weights

        Returns
        -------
        tangent_updates : array of shape (n_prototypes, *point_shape)
        """
        def log_one(w, x):
            return self.manifold.log_map(w, x)

        # For each prototype, compute log maps from all data points
        def update_one_proto(w, h_col):
            # log_map(w, x_i) for all x_i: (n_samples, *point_shape)
            tangents = jax.vmap(log_one, in_axes=(None, 0))(w, X)
            # Weighted mean of tangent vectors
            weighted = h_col[:, None, None] * tangents
            return jnp.mean(weighted, axis=0)

        # vmap over prototypes
        return jax.vmap(update_one_proto)(prototypes, h.T)

    def _apply_updates(self, prototypes, tangent_updates, lr):
        """Apply exponential map to move prototypes along tangent vectors.

        Includes safety bound to stay within injectivity radius (Eq. 7).

        Parameters
        ----------
        prototypes : array of shape (n_prototypes, *point_shape)
        tangent_updates : array of shape (n_prototypes, *point_shape)
        lr : float

        Returns
        -------
        new_prototypes : array of shape (n_prototypes, *point_shape)
        """
        def step_one(w, v):
            scaled_v = lr * v

            # Safety bound (Eq. 7): clip to injectivity radius
            inj = self.manifold.injectivity_radius(w)
            v_norm = jnp.linalg.norm(scaled_v, 'fro')
            scale = jnp.where(
                v_norm > self.tau * inj,
                self.tau * inj / (v_norm + 1e-10),
                1.0
            )
            safe_v = scaled_v * scale

            # Exponential map: move on manifold
            w_new = self.manifold.exp_map(w, safe_v)
            # Project to ensure manifold membership
            return self.manifold.project(w_new)

        return jax.vmap(step_one)(prototypes, tangent_updates)

    def fit(self, X):
        """Fit Riemannian Neural Gas.

        Parameters
        ----------
        X : array of shape (n_samples, *point_shape)
            Data points on the manifold.

        Returns
        -------
        self
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples = X.shape[0]

        # Initialize prototypes from data
        key = self.key
        indices = jax.random.choice(key, n_samples,
                                    (self.n_prototypes,), replace=False)
        prototypes = X[indices]

        lambda_init_val = (self.lambda_init if self.lambda_init
                           else self.n_prototypes / 2.0)

        loss_history = []

        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac
            lam_t = lambda_init_val * (self.lambda_final / lambda_init_val) ** frac

            # Geodesic distance matrix: (n_samples, n_prototypes)
            distances = self._compute_distance_matrix(X, prototypes)

            # NG ranking
            order = jnp.argsort(distances, axis=1)
            ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
            h = jnp.exp(-ranks / lam_t)

            # Compute weighted tangent updates
            tangent_updates = self._compute_updates(X, prototypes, h)

            # Apply exponential map with safety bound
            prototypes = self._apply_updates(prototypes, tangent_updates, lr_t)

            # Energy
            energy = float(jnp.sum(h * distances))
            loss_history.append(energy)

            # Convergence check
            if t > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        self.prototypes_ = prototypes
        self.n_iter_ = t + 1
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        return self

    def predict(self, X):
        """Assign each point to the nearest prototype (BMU).

        Parameters
        ----------
        X : array of shape (n_samples, *point_shape)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        distances = self._compute_distance_matrix(X, self.prototypes_)
        return jnp.argmin(distances, axis=1)

    def transform(self, X):
        """Compute geodesic distance matrix to all prototypes.

        Parameters
        ----------
        X : array of shape (n_samples, *point_shape)

        Returns
        -------
        distances : array of shape (n_samples, n_prototypes)
            Geodesic distances (not squared).
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return jnp.sqrt(self._compute_distance_matrix(X, self.prototypes_))

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'lr_init': self.lr_init,
            'lr_final': self.lr_final,
            'lambda_final': self.lambda_final,
            'tau': self.tau,
        })
        if self.lambda_init is not None:
            hp['lambda_init'] = self.lambda_init
        # Store manifold type and params for reconstruction
        manifold = self.manifold
        hp['manifold_type'] = type(manifold).__name__
        if hasattr(manifold, 'n'):
            hp['manifold_n'] = manifold.n
        if hasattr(manifold, 'k'):
            hp['manifold_k'] = manifold.k
        return hp

    def _get_fitted_arrays(self):
        arrays = {}
        if self.prototypes_ is not None:
            arrays['prototypes_'] = np.asarray(self.prototypes_)
        if self.loss_history_ is not None:
            arrays['loss_history_'] = np.asarray(self.loss_history_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        if 'prototypes_' in arrays:
            self.prototypes_ = jnp.asarray(arrays['prototypes_'])
        if 'loss_history_' in arrays:
            self.loss_history_ = jnp.asarray(arrays['loss_history_'])

    @classmethod
    def _reconstruct_manifold(cls, hp):
        """Reconstruct manifold from saved hyperparameters."""
        from prosemble.core.manifolds import SO, SPD, Grassmannian
        mtype = hp.get('manifold_type', '')
        if mtype == 'SO':
            return SO(int(hp['manifold_n']))
        elif mtype == 'SPD':
            return SPD(int(hp['manifold_n']))
        elif mtype == 'Grassmannian':
            return Grassmannian(int(hp['manifold_n']), int(hp['manifold_k']))
        else:
            raise ValueError(f"Unknown manifold type: {mtype}")

    @classmethod
    def load(cls, path):
        """Load a saved RiemannianNeuralGas model."""
        import json
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data['__metadata__']))
        hp = dict(metadata['hyperparams'])  # copy to avoid mutation

        manifold = cls._reconstruct_manifold(hp)

        # Remove manifold-specific keys
        hp.pop('manifold_type', None)
        hp.pop('manifold_n', None)
        hp.pop('manifold_k', None)

        model = cls(manifold=manifold, **hp)

        # Restore fitted arrays
        fitted_keys = metadata.get('fitted_array_names', [])
        arrays = {k: data[k] for k in fitted_keys if k in data.files}
        model._set_fitted_arrays(arrays)

        model.n_iter_ = metadata.get('n_iter_', 0)
        model.loss_ = metadata.get('loss_', None)

        return model

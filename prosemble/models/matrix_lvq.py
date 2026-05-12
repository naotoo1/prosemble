"""
Generalized Matrix LVQ (GMLVQ).

GLVQ with a learned linear transformation Omega that maps data into
a discriminative subspace: d(x, w) = ||Omega(x - w)||^2.

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init


@jit
def _predict_gmlvq_jit(X, prototypes, omega, proto_labels):
    """JIT-compiled GMLVQ prediction with learned Omega metric."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


class GMLVQ(SupervisedPrototypeModel):
    """Generalized Matrix Learning Vector Quantization.

    Learns a global linear mapping Omega (d x latent_dim) such that
    distances are computed in the transformed space::

        d(x, w) = (x - w)^T Omega^T Omega (x - w)

    The relevance matrix Lambda = Omega^T Omega captures feature
    correlations.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the latent space. If None, uses input dim.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    beta : float
        Transfer function steepness.
    """

    def __init__(self, latent_dim=None, beta=10.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.omega_ = None

    def _get_resume_params(self, params):
        params['omega'] = self.omega_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        omega = identity_omega_init(n_features, latent_dim)

        params = {'prototypes': prototypes, 'omega': omega}
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        omega = params['omega']
        # Omega distance: d(x, w) = ||Omega(x - w)||^2
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,dl->npl', diff, omega)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        from prosemble.core.losses import glvq_loss_with_transfer
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_ = params['omega']

    @property
    def omega_matrix(self):
        """Return the learned Omega matrix."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return Lambda = Omega^T Omega (relevance matrix)."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_.T @ self.omega_

    def predict(self, X):
        """Predict using learned Omega distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_gmlvq_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omega_ is not None:
            attrs.append('omega_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_ is not None:
            arrays['omega_'] = np.asarray(self.omega_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_' in arrays:
            self.omega_ = jnp.asarray(arrays['omega_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp

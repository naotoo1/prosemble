"""
One-Class GMLVQ (OC-GMLVQ).

Extends OC-GLVQ with a global Omega matrix (GMLVQ-style metric adaptation).
The distance becomes:

    d_Ω(x, w_k) = ||Ω(x - w_k)||²

where Ω is a learned (d × l) projection matrix. The implicit metric
Λ = Ω^T Ω captures feature correlations for one-class classification.

References
----------
.. [1] Sato, A., & Yamada, K. (1995). Generalized Learning Vector
       Quantization. NIPS.
.. [2] Schneider, Biehl, Hammer (2009). Adaptive Relevance Matrices
       in Learning Vector Quantization. Neural Computation, 21(12).
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.oc_glvq import OCGLVQ
from prosemble.core.initializers import identity_omega_init
from prosemble.core.activations import sigmoid_beta


class OCGMLVQ(OCGLVQ):
    """One-Class GMLVQ with global Omega projection.

    Learns a global linear projection Ω that captures feature
    correlations for one-class classification.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the projected space. Default: n_features.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned projection matrix.
    """

    def __init__(self, latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.omega_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omega'] = self.omega_
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        latent_dim = self.latent_dim if self.latent_dim is not None else n_features
        params['omega'] = identity_omega_init(n_features, latent_dim)
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=params['prototypes'],
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        thetas = params['thetas']
        omega = params['omega']

        # Omega-projected squared distances
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, K, d)
        projected = jnp.einsum('nkd,dl->nkl', diff, omega)  # (n, K, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, K)

        # OC-GLVQ mu
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = thetas[nearest_idx]
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s * (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        transfer = self.transfer_fn or sigmoid_beta
        return jnp.mean(transfer(mu + self.margin, self.beta))

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omega_ = params['omega']

    def decision_function(self, X):
        """Compute scores using Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, self.omega_)
        distances = jnp.sum(projected ** 2, axis=2)
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = self.thetas_[nearest_idx]
        mu = (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        return 1.0 - jax.nn.sigmoid(self.beta * mu)

    @property
    def omega_matrix(self):
        """Return the learned projection matrix Ω."""
        if self.omega_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return the implicit metric Λ = Ω^T Ω."""
        return self.omega_matrix.T @ self.omega_matrix

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
        hp['latent_dim'] = self.latent_dim
        return hp

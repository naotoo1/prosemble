"""
Generalized Tangent LVQ (GTLVQ).

Each prototype has a tangent subspace defined by an orthogonal basis
Omega_k. The tangent distance projects out the tangent directions.

References
----------
.. [1] Saralajew, S., & Villmann, T. (2016). Adaptive tangent
       distances in generalized learning vector quantization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import random_omega_init
from prosemble.core.utils import orthogonalize


@jit
def _predict_gtlvq_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled GTLVQ prediction with tangent distance."""
    diff = X[:, None, :] - prototypes[None, :, :]
    proj = jnp.einsum('npd,pds->nps', diff, omegas)
    recon = jnp.einsum('nps,pds->npd', proj, omegas)
    tang_diff = diff - recon
    distances = jnp.sum(tang_diff ** 2, axis=2)
    return wtac(distances, proto_labels)


class GTLVQ(SupervisedPrototypeModel):
    """Generalized Tangent Learning Vector Quantization.

    Each prototype k has a subspace basis Omega_k. The tangent
    distance is: d(x, w_k) = ||P_k(x - w_k)||^2, where
    P_k = I - Omega_k @ Omega_k^T is the orthogonal projector.

    Parameters
    ----------
    subspace_dim : int
        Dimension of each prototype's tangent subspace.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    beta : float
        Transfer function steepness.
    transfer_fn : callable, optional
        Transfer function for loss shaping. Default: identity.
    margin : float
        Margin added to the loss. Default: 0.0.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, subspace_dim=2, beta=10.0, **kwargs):
        super().__init__(**kwargs)
        self.subspace_dim = subspace_dim
        self.beta = beta
        self.omegas_ = None

    def _get_resume_params(self, params):
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        n_protos = prototypes.shape[0]

        # Initialize each Omega as random orthogonal
        keys = jax.random.split(key2, n_protos)
        omegas = jnp.stack([
            random_omega_init(n_features, self.subspace_dim, k) for k in keys
        ])  # (p, d, subspace_dim)

        params = {'prototypes': prototypes, 'omegas': omegas}
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
        omegas = params['omegas']  # (p, d, s)
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        # Projector P_k = I - Omega_k @ Omega_k^T
        # P_k @ diff_k = diff_k - Omega_k @ (Omega_k^T @ diff_k)
        # projected = diff - omegas @ (omegas^T @ diff)
        proj_onto_subspace = jnp.einsum('npd,pds->nps', diff, omegas)  # (n, p, s)
        reconstruction = jnp.einsum('nps,pds->npd', proj_onto_subspace, omegas)  # (n, p, d)
        tangent_diff = diff - reconstruction  # (n, p, d)
        distances = jnp.sum(tangent_diff ** 2, axis=2)  # (n, p)
        from prosemble.core.losses import glvq_loss_with_transfer
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _post_update(self, params):
        """Re-orthogonalize Omega matrices via polar decomposition."""
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_gtlvq_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omegas_ is not None:
            attrs.append('omegas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omegas_ is not None:
            arrays['omegas_'] = np.asarray(self.omegas_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['subspace_dim'] = self.subspace_dim
        hp['beta'] = self.beta
        return hp

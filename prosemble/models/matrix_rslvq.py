"""
Matrix Robust Soft LVQ (MRSLVQ) and Localized Matrix RSLVQ (LMRSLVQ).

RSLVQ with learned linear transformation(s) Omega for metric adaptation.
MRSLVQ uses a single global Omega; LMRSLVQ uses per-prototype Omega_k.

Distance: d(x, w_k) = (x - w_k)^T Omega^T Omega (x - w_k)
Loss: -log(P(correct_class|x) / P(all|x))  [RSLVQ objective]

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init
from prosemble.core.losses import rslvq_loss
from prosemble.core.pooling import stratified_min_pooling


@jit
def _predict_mrslvq_jit(X, prototypes, omega, proto_labels):
    """JIT-compiled MRSLVQ prediction with learned Omega metric."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(4,))
def _predict_proba_mrslvq_jit(X, prototypes, omega, proto_labels, n_classes):
    """JIT-compiled MRSLVQ probability prediction."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


@jit
def _predict_lmrslvq_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled LMRSLVQ prediction with per-prototype Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(4,))
def _predict_proba_lmrslvq_jit(X, prototypes, omegas, proto_labels, n_classes):
    """JIT-compiled LMRSLVQ probability prediction."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


class MRSLVQ(SupervisedPrototypeModel):
    """Matrix Robust Soft Learning Vector Quantization.

    Combines the RSLVQ probabilistic loss with a learned global linear
    mapping Omega (d x latent_dim) for metric adaptation::

        d(x, w) = (x - w)^T Omega^T Omega (x - w)

    The relevance matrix Lambda = Omega^T Omega captures feature
    correlations in the probabilistic framework.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
    latent_dim : int, optional
        Dimensionality of the latent space. If None, uses input dim.
    rejection_confidence : float, optional
        Minimum class probability for a confident prediction (0 to 1).
        Samples below this threshold are rejected (labeled -1) when using
        ``predict_with_rejection()``. Default is None (no rejection).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, sigma=1.0, latent_dim=None, rejection_confidence=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.rejection_confidence = rejection_confidence
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
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,dl->npl', diff, omega)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        return rslvq_loss(distances, y, proto_labels, sigma=self.sigma)

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
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
        return _predict_mrslvq_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict class probabilities using Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_mrslvq_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_,
            self.n_classes_
        )

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected / "I don't know").

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels, or -1 for rejected samples.
        """
        self._check_fitted()
        threshold = (confidence if confidence is not None
                     else self.rejection_confidence)
        if threshold is None:
            return self.predict(X)

        X = jnp.asarray(X, dtype=jnp.float32)
        proba = self.predict_proba(X)
        max_proba = jnp.max(proba, axis=1)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(max_proba >= threshold, preds, -1)

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
        hp['sigma'] = self.sigma
        hp['rejection_confidence'] = self.rejection_confidence
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp


class LMRSLVQ(SupervisedPrototypeModel):
    """Localized Matrix Robust Soft Learning Vector Quantization.

    Each prototype k has its own Omega_k matrix. The distance from
    sample x to prototype w_k is::

        d(x, w_k) = (x - w_k)^T Omega_k^T Omega_k (x - w_k)

    Combined with the RSLVQ probabilistic loss for metric-adaptive
    soft classification with local relevance learning.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
    latent_dim : int, optional
        Latent space dimensionality per prototype. If None, uses input dim.
    rejection_confidence : float, optional
        Minimum class probability for a confident prediction (0 to 1).
        Samples below this threshold are rejected (labeled -1) when using
        ``predict_with_rejection()``. Default is None (no rejection).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, sigma=1.0, latent_dim=None, rejection_confidence=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.rejection_confidence = rejection_confidence
        self.omegas_ = None

    def _get_resume_params(self, params):
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        n_protos = prototypes.shape[0]
        omega_single = identity_omega_init(n_features, latent_dim)
        omegas = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))

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
        omegas = params['omegas']  # (p, d, l)
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,pdl->npl', diff, omegas)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        return rslvq_loss(distances, y, proto_labels, sigma=self.sigma)

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omegas_ = params['omegas']

    def predict(self, X):
        """Predict using local Omega distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_lmrslvq_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict class probabilities using local Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_lmrslvq_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_,
            self.n_classes_
        )

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected / "I don't know").

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels, or -1 for rejected samples.
        """
        self._check_fitted()
        threshold = (confidence if confidence is not None
                     else self.rejection_confidence)
        if threshold is None:
            return self.predict(X)

        X = jnp.asarray(X, dtype=jnp.float32)
        proba = self.predict_proba(X)
        max_proba = jnp.max(proba, axis=1)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(max_proba >= threshold, preds, -1)

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
        hp['sigma'] = self.sigma
        hp['rejection_confidence'] = self.rejection_confidence
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp

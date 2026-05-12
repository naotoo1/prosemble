"""
Matrix RSLVQ with Neural Gas Cooperation (MRSLVQ_NG, LMRSLVQ_NG).

Combines RSLVQ's probabilistic soft-assignment with Neural Gas
neighborhood cooperation and learned linear metric adaptation.

MRSLVQ_NG: global Omega matrix + NG cooperation
LMRSLVQ_NG: per-prototype Omega matrices + NG cooperation

When gamma -> 0, recovers standard MRSLVQ / LMRSLVQ behavior.

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks.
.. [3] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init
from prosemble.core.losses import ng_rslvq_loss
from prosemble.core.pooling import stratified_min_pooling


@jit
def _predict_mrslvq_ng_jit(X, prototypes, omega, proto_labels):
    """JIT-compiled MRSLVQ_NG prediction with learned Omega metric."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(4,))
def _predict_proba_mrslvq_ng_jit(X, prototypes, omega, proto_labels, n_classes):
    """JIT-compiled MRSLVQ_NG probability prediction."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


@jit
def _predict_lmrslvq_ng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled LMRSLVQ_NG prediction with per-prototype Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(4,))
def _predict_proba_lmrslvq_ng_jit(X, prototypes, omegas, proto_labels, n_classes):
    """JIT-compiled LMRSLVQ_NG probability prediction."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


class MRSLVQ_NG(SupervisedPrototypeModel):
    """Matrix Robust Soft LVQ with Neural Gas Cooperation.

    Combines:

    - RSLVQ probabilistic loss: ``-log(P(correct|x))``
    - Neural Gas cooperation: all prototypes weighted by rank via
      ``exp(-rank / gamma)``
    - Global Omega matrix for metric adaptation:
      ``d(x, w) = (x - w)^T Omega^T Omega (x - w)``

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
    latent_dim : int, optional
        Dimensionality of the latent space. If None, uses input dim.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay. Default: computed from max_iter.
    rejection_confidence : float, optional
        Minimum class probability for confident prediction (0 to 1).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, sigma=1.0, latent_dim=None, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None,
                 rejection_confidence=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.rejection_confidence = rejection_confidence
        self.omega_ = None
        self.gamma_ = None

        # Freeze gamma from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omega'] = self.omega_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        omega = identity_omega_init(n_features, latent_dim)

        # Compute gamma_init from prototype count
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)

        gamma_init = (self.gamma_init if self.gamma_init is not None
                      else max_per_class / 2.0)
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (
                self.gamma_final / gamma_init
            ) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omega': omega,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
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
        gamma = params['gamma']
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,dl->npl', diff, omega)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        return ng_rslvq_loss(distances, y, proto_labels,
                             sigma=self.sigma, gamma=gamma)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omega_ = params['omega']
        self.gamma_ = float(params['gamma'])

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
        return _predict_mrslvq_ng_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict class probabilities using Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_mrslvq_ng_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_,
            self.n_classes_
        )

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
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
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_' in arrays:
            self.omega_ = jnp.asarray(arrays['omega_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['rejection_confidence'] = self.rejection_confidence
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp


class LMRSLVQ_NG(SupervisedPrototypeModel):
    """Localized Matrix Robust Soft LVQ with Neural Gas Cooperation.

    Each prototype k has its own Omega_k matrix. Combined with RSLVQ
    probabilistic loss and NG rank-based neighborhood cooperation.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
    latent_dim : int, optional
        Latent space dimensionality per prototype. If None, uses input dim.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay. Default: computed from max_iter.
    rejection_confidence : float, optional
        Minimum class probability for confident prediction (0 to 1).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, sigma=1.0, latent_dim=None, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None,
                 rejection_confidence=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.rejection_confidence = rejection_confidence
        self.omegas_ = None
        self.gamma_ = None

        # Freeze gamma from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omegas'] = self.omegas_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
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

        # Compute gamma_init from prototype count
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)

        gamma_init = (self.gamma_init if self.gamma_init is not None
                      else max_per_class / 2.0)
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (
                self.gamma_final / gamma_init
            ) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omegas': omegas,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
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
        gamma = params['gamma']
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,pdl->npl', diff, omegas)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        return ng_rslvq_loss(distances, y, proto_labels,
                             sigma=self.sigma, gamma=gamma)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omegas_ = params['omegas']
        self.gamma_ = float(params['gamma'])

    def predict(self, X):
        """Predict using local Omega distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_lmrslvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict class probabilities using local Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_lmrslvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_,
            self.n_classes_
        )

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
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
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['rejection_confidence'] = self.rejection_confidence
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp

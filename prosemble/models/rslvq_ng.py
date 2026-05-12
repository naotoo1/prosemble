"""
RSLVQ with Neural Gas Cooperation (RSLVQ_NG).

Combines RSLVQ's probabilistic soft-assignment with Neural Gas
neighborhood cooperation. All prototypes contribute to the loss
via Gaussian mixture probabilities, modulated by NG rank-based
neighborhood weights.

When gamma -> 0, only the nearest prototype matters, recovering
standard RSLVQ behavior.

References
----------
.. [1] Seo, S., & Obermayer, K. (2007). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import ng_rslvq_loss


class RSLVQ_NG(SupervisedPrototypeModel):
    """Robust Soft LVQ with Neural Gas Cooperation.

    Combines:

    - RSLVQ probabilistic loss: ``-log(P(correct|x))``
    - Neural Gas cooperation: all prototypes weighted by rank via
      ``exp(-rank / gamma)``
    - Euclidean distance

    The NG neighborhood modulates RSLVQ's Gaussian probabilities,
    emphasizing nearby prototypes. Gamma decays during training from
    gamma_init to gamma_final.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
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

    def __init__(self, sigma=1.0, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, rejection_confidence=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.rejection_confidence = rejection_confidence
        self.gamma_ = None

        # Freeze gamma from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

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
        gamma = params['gamma']
        distances = self.distance_fn(X, prototypes)
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
        self.gamma_ = float(params['gamma'])

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

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['rejection_confidence'] = self.rejection_confidence
        return hp

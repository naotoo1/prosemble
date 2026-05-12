"""
Generalized Relevance LVQ (GRLVQ).

GLVQ with per-feature relevance weighting. Learns which features
are most important for classification.

References
----------
.. [1] Hammer, B., & Villmann, T. (2002). Generalized relevance
       learning vector quantization. Neural Networks.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel


class GRLVQ(SupervisedPrototypeModel):
    """Generalized Relevance Learning Vector Quantization.

    Learns per-feature relevance weights lambda_j such that the
    weighted distance is: d(x, w) = sum_j lambda_j * (x_j - w_j)^2

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    beta : float
        Transfer function steepness parameter.
    transfer_fn : callable, optional
        Transfer function for loss shaping. Default: identity.
    margin : float
        Margin added to the loss. Default: 0.0.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, beta=10.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.relevances_ = None

    def _get_resume_params(self, params):
        params['relevances'] = self.relevances_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        # Initialize uniform relevances
        relevances = jnp.ones(n_features) / n_features
        params = {'prototypes': prototypes, 'relevances': relevances}
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
        relevances = params['relevances']
        # Weighted squared Euclidean distance
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        # Relevances applied after softmax to ensure positivity & normalization
        lam = jax.nn.softmax(relevances)
        distances = jnp.sum(lam[None, None, :] * diff ** 2, axis=2)  # (n, p)
        from prosemble.core.losses import glvq_loss_with_transfer
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _post_update(self, params):
        # No explicit constraint needed since we use softmax in loss
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.relevances_ = jax.nn.softmax(params['relevances'])

    @property
    def relevance_profile(self):
        """Return the learned relevance weights (normalized)."""
        if self.relevances_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.relevances_

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.relevances_ is not None:
            attrs.append('relevances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.relevances_ is not None:
            arrays['relevances_'] = np.asarray(self.relevances_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        return hp

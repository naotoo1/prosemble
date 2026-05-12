"""
Classification-By-Components (CBC).

A reasoning-based classification model where components (prototypes
without class labels) learn both detection (similarity) and reasoning
(positive/negative evidence per class).

References
----------
.. [1] Saralajew, S., Holdijk, L., & Villmann, T. (2020).
       Classification-by-Components: Probabilistic Modeling of
       Reasoning over a Set of Components. NeurIPS.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel, NotFittedError
from prosemble.core.competitions import cbcc
from prosemble.core.losses import margin_loss
from prosemble.core.similarities import gaussian_similarity


@jit
def _predict_proba_cbc_jit(X, components, reasonings, sigma_sq):
    """JIT-compiled CBC probability prediction."""
    diff = X[:, None, :] - components[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=2)
    detections = gaussian_similarity(dist_sq, variance=sigma_sq)
    return cbcc(detections, reasonings)


@jit
def _predict_cbc_jit(X, components, reasonings, sigma_sq):
    """JIT-compiled CBC class prediction."""
    probs = _predict_proba_cbc_jit(X, components, reasonings, sigma_sq)
    return jnp.argmax(probs, axis=1)


class CBC(SupervisedPrototypeModel):
    """Classification-By-Components.

    Components detect patterns in the input (via similarity), then
    reasoning matrices determine how each detection contributes
    evidence for/against each class.

    Parameters
    ----------
    n_components : int
        Number of components (analogous to prototypes, but classless).
    n_classes : int
        Number of output classes.
    sigma : float
        Bandwidth for Gaussian similarity.
    margin : float
        Margin for the margin loss.
    components_initializer : callable, optional
        Initializer for component vectors. Default: None.
    reasonings_initializer : callable, optional
        Initializer for reasoning matrix. Default: None.
    similarity_fn : callable, optional
        Similarity function for component detection. Default: None
        (uses Gaussian similarity).
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, n_components=5, n_classes=2, sigma=1.0,
                 margin=0.3, components_initializer=None,
                 reasonings_initializer=None, similarity_fn=None, **kwargs):
        kwargs['margin'] = margin
        super().__init__(**kwargs)
        self.n_components = n_components
        self._n_classes = n_classes
        self.sigma = sigma
        self.components_initializer = components_initializer
        self.reasonings_initializer = reasonings_initializer
        if similarity_fn is not None:
            self._similarity_fn = similarity_fn
        else:
            self._similarity_fn = gaussian_similarity
        self.components_ = None
        self.reasonings_ = None

    def _get_resume_params(self, params):
        # CBC uses 'components' key instead of 'prototypes'
        return {'components': self.components_, 'reasonings': self.reasonings_}

    def _init_state(self, X, y, key):
        n_classes = self._n_classes
        key1, key2 = jax.random.split(key)

        # Initialize components
        if self.components_initializer is not None:
            components = self.components_initializer(X, key1, self.n_components)
        else:
            indices = jax.random.choice(key1, X.shape[0], (self.n_components,), replace=False)
            components = X[indices]

        # Initialize reasoning matrices: (n_components, n_classes, 2)
        if self.reasonings_initializer is not None:
            reasonings = self.reasonings_initializer(
                self.n_components, n_classes, key2
            )
        else:
            # Default: near uniform p=0.5, n=0.5 with small noise
            reasonings = jnp.ones((self.n_components, n_classes, 2)) * 0.5
            reasonings = reasonings + 0.01 * jax.random.normal(key2, reasonings.shape)

        params = {'components': components, 'reasonings': reasonings}
        opt_state = self._optimizer.init(params)

        # proto_labels not meaningful for CBC, but needed by base class
        proto_labels = jnp.zeros(self.n_components, dtype=jnp.int32)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=components,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        components = params['components']
        reasonings = params['reasonings']

        # Compute similarities (detections)
        diff = X[:, None, :] - components[None, :, :]  # (n, c, d)
        dist_sq = jnp.sum(diff ** 2, axis=2)  # (n, c)
        detections = gaussian_similarity(dist_sq, variance=self.sigma ** 2)

        # Compute class probabilities via CBC reasoning
        class_probs = cbcc(detections, reasonings)  # (n, n_classes)

        # Margin loss
        y_one_hot = jax.nn.one_hot(y, self._n_classes)
        return margin_loss(class_probs, y_one_hot, margin=self.margin)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        best_params = kwargs.get('best_params')
        if getattr(self, 'restore_best', False) and best_params is not None:
            params = best_params
        self.prototypes_ = params['components']
        self.components_ = params['components']
        self.reasonings_ = params['reasonings']
        self.prototype_labels_ = proto_labels
        self.loss_ = float(loss_history[-1]) if len(loss_history) > 0 else None
        self.loss_history_ = jnp.array(loss_history)
        self.n_iter_ = n_iter
        val_loss_history = kwargs.get('val_loss_history')
        if val_loss_history is not None:
            self.val_loss_history_ = jnp.array(val_loss_history)
        best_loss = kwargs.get('best_loss')
        if best_loss is not None:
            self.best_loss_ = best_loss

    def predict(self, X):
        """Predict class labels."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_cbc_jit(
            X, self.components_, self.reasonings_, self.sigma ** 2
        )

    def predict_proba(self, X):
        """Predict class probabilities via CBC reasoning."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_cbc_jit(
            X, self.components_, self.reasonings_, self.sigma ** 2
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.reasonings_ is not None:
            attrs.append('reasonings_')
        if self.components_ is not None:
            attrs.append('components_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.reasonings_ is not None:
            arrays['reasonings_'] = np.asarray(self.reasonings_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'reasonings_' in arrays:
            self.reasonings_ = jnp.asarray(arrays['reasonings_'])
        if self.prototypes_ is not None:
            self.components_ = self.prototypes_

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['n_components'] = self.n_components
        hp['n_classes'] = self._n_classes
        hp['sigma'] = self.sigma
        return hp

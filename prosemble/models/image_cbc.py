"""
Image CBC — Classification-By-Components with a CNN backbone.

Components (classless prototypes) and inputs are both passed through
the same CNN, then detection similarity and reasoning are computed
in the latent space.

Architecture::

    Image (H,W,C) ---> CNN ---> latent_x
                                    |
                                    v
    Component (H,W,C) -> CNN -> latent_c    similarity(latent_x, latent_c)
                                                       |
                                                       v
                                              reasoning -> class_probs
                                                       |
                                                       v
                                                  margin loss

References
----------
.. [1] Saralajew, S., Holdijk, L., & Villmann, T. (2020).
       Classification-by-Components: Probabilistic Modeling of
       Reasoning over a Set of Components. NeurIPS.
"""

import jax
import jax.numpy as jnp

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import cbcc
from prosemble.core.losses import margin_loss
from prosemble.core.similarities import gaussian_similarity
from prosemble.models.lvqmln import _cnn_init, _cnn_forward


class ImageCBC(SupervisedPrototypeModel):
    """Image CBC — CBC with a CNN embedding network.

    Both input images and component images pass through the same CNN
    backbone. Detection similarity and reasoning matrices then produce
    class probabilities.

    Parameters
    ----------
    input_shape : tuple
        (height, width, channels) of input images.
    channels : list of int
        CNN output channels per layer.
    kernel_sizes : list of int
        Kernel sizes per conv layer.
    latent_dim : int
        CNN embedding dimension.
    n_components : int
        Number of components (classless prototypes).
    n_classes : int
        Number of output classes.
    sigma : float
        Bandwidth for Gaussian similarity.
    margin : float
        Margin for the margin loss.
    activation : str
        CNN activation.
    max_iter : int
        Training epochs.
    lr : float
        Learning rate.
    """

    def __init__(self, input_shape=(28, 28, 1), channels=None,
                 kernel_sizes=None, latent_dim=32,
                 n_components=5, n_classes=2, sigma=1.0,
                 activation='relu', **kwargs):
        kwargs.setdefault('margin', 0.3)
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.channels = channels or [16, 32]
        self.kernel_sizes = kernel_sizes or [3, 3]
        self.latent_dim = latent_dim
        self.n_components = n_components
        self._n_classes = n_classes
        self.sigma = sigma
        self.activation = activation
        self.backbone_params_ = None
        self.components_ = None
        self.reasonings_ = None

    def _get_resume_params(self, params):
        # ImageCBC uses 'components' key instead of 'prototypes'
        return {
            'components': self.components_,
            'reasonings': self.reasonings_,
            'backbone': self.backbone_params_,
        }

    def _init_state(self, X, y, key):
        key1, key2, key3 = jax.random.split(key, 3)

        backbone_params = _cnn_init(
            key1, self.input_shape, self.channels,
            self.kernel_sizes, self.latent_dim, self.activation,
        )

        # Components as images
        X_flat = X.reshape(X.shape[0], -1)
        indices = jax.random.choice(key2, X.shape[0], (self.n_components,), replace=False)
        components = X_flat[indices].reshape(-1, *self.input_shape)

        # Reasoning matrices: (n_components, n_classes, 2)
        reasonings = jnp.ones((self.n_components, self._n_classes, 2)) * 0.5
        reasonings = reasonings + 0.01 * jax.random.normal(key3, reasonings.shape)

        params = {
            'components': components,
            'reasonings': reasonings,
            'backbone': backbone_params,
        }
        opt_state = self._optimizer.init(params)

        # proto_labels not meaningful for CBC
        proto_labels = jnp.zeros(self.n_components, dtype=jnp.int32)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=components.reshape(-1, int(jnp.prod(jnp.array(self.input_shape)))),
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        backbone = params['backbone']
        components = params['components']
        reasonings = params['reasonings']

        X_img = X.reshape(-1, *self.input_shape)
        comp_img = components.reshape(-1, *self.input_shape)

        latent_x = _cnn_forward(backbone, X_img, self.activation)
        latent_c = _cnn_forward(backbone, comp_img, self.activation)

        # Similarity in latent space
        diff = latent_x[:, None, :] - latent_c[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=2)
        detections = gaussian_similarity(dist_sq, variance=self.sigma ** 2)

        # CBC reasoning
        class_probs = cbcc(detections, reasonings)

        # Margin loss
        y_one_hot = jax.nn.one_hot(y, self._n_classes)
        return margin_loss(class_probs, y_one_hot, margin=self.margin)

    def _post_update(self, params):
        """Clamp component images to [0, 1] for valid pixel values."""
        if 'components' in params:
            components = jnp.clip(params['components'], 0.0, 1.0)
            return {**params, 'components': components}
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        best_params = kwargs.get('best_params')
        if getattr(self, 'restore_best', False) and best_params is not None:
            params = best_params
        self.prototypes_ = params['components']
        self.components_ = params['components']
        self.reasonings_ = params['reasonings']
        self.backbone_params_ = params['backbone']
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
        proba = self.predict_proba(X)
        return jnp.argmax(proba, axis=1)

    def predict_proba(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        comp_img = self.components_.reshape(-1, *self.input_shape)

        latent_x = _cnn_forward(self.backbone_params_, X_img, self.activation)
        latent_c = _cnn_forward(self.backbone_params_, comp_img, self.activation)

        diff = latent_x[:, None, :] - latent_c[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=2)
        detections = gaussian_similarity(dist_sq, variance=self.sigma ** 2)
        return cbcc(detections, self.reasonings_)

    def transform(self, X):
        """Transform images to latent space."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _cnn_forward(
            self.backbone_params_, X.reshape(-1, *self.input_shape), self.activation
        )

    def _check_fitted(self):
        if self.components_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'input_shape': list(self.input_shape),
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'latent_dim': self.latent_dim,
            'n_components': self.n_components,
            'n_classes': self._n_classes,
            'sigma': self.sigma,
            'activation': self.activation,
        })
        return hp

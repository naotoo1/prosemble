"""
Probabilistic LVQ (PLVQ).

Combines a learned nonlinear transformation (MLP backbone) with
probabilistic soft assignment (Gaussian mixture) for classification.

Like LVQMLN, the backbone transforms inputs into a latent space
where prototypes reside. Unlike LVQMLN (which uses GLVQ's hard
winner-take-all loss), PLVQ uses a probabilistic loss based on
Gaussian mixtures — making it a deep variant of RSLVQ.

Architecture::

    Input (d) -> MLP -> Latent (latent_dim)
                          |
                          v
                distance(latent_x, prototypes)
                          |
                          v
              Gaussian mixture class probabilities
                          |
                          v
              -log P(correct class | x)

References
----------
.. [1] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation.
.. [2] Villmann, T., et al. (2017). Prototype-based Neural Network
       Layers: Incorporating Vector Quantization. arXiv:1812.01214.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import rslvq_loss, nllr_loss
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.models.lvqmln import _mlp_init, _mlp_forward


class PLVQ(SupervisedPrototypeModel):
    """Probabilistic LVQ with learned nonlinear transformation.

    Combines an MLP backbone (learned metric) with probabilistic
    soft assignment via Gaussian mixtures. The loss is the negative
    log-likelihood of the correct class:

        p(k|x) = exp(-d(f(x), w_k)² / 2σ²) / Z
        P(class|x) = Σ_{k∈class} p(k|x)
        loss = -log(P(correct|x) / P(all|x))

    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes.
    latent_dim : int
        Latent space dimension.
    activation : str
        Activation: 'sigmoid', 'relu', 'tanh', 'leaky_relu', 'selu'.
    sigma : float
        Bandwidth of the Gaussian mixture.
    loss_type : str
        'rslvq' (robust, default) or 'nllr' (likelihood ratio).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, hidden_sizes=None, latent_dim=2,
                 activation='sigmoid', sigma=1.0, loss_type='rslvq',
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
            optimizer=optimizer, transfer_fn=transfer_fn, margin=margin,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
        )
        self.hidden_sizes = hidden_sizes or [10]
        self.latent_dim = latent_dim
        self.activation = activation
        self.sigma = sigma
        self.loss_type = loss_type
        self.backbone_params_ = None

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        return params

    def _init_state(self, X, y, key):
        """Initialize backbone + prototypes in latent space."""
        n_features = X.shape[1]

        layer_sizes = [n_features] + list(self.hidden_sizes) + [self.latent_dim]

        key1, key2 = jax.random.split(key)

        # Initialize backbone
        backbone_params = _mlp_init(key1, layer_sizes, self.activation)

        # Project data for prototype init
        latent_X = _mlp_forward(backbone_params, X, self.activation)
        prototypes, proto_labels = self._init_prototypes(
            latent_X, y, self.n_prototypes_per_class, key2
        )

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
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
        """Probabilistic loss in latent space."""
        latent_x = _mlp_forward(params['backbone'], X, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, params['prototypes'])
        if self.loss_type == 'rslvq':
            return rslvq_loss(distances, y, proto_labels, sigma=self.sigma)
        elif self.loss_type == 'nllr':
            return nllr_loss(distances, y, proto_labels, sigma=self.sigma)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        """Store backbone params alongside prototypes."""
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']

    def predict(self, X):
        """Predict via most probable class."""
        proba = self.predict_proba(X)
        return jnp.argmax(proba, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities via Gaussian mixture."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, self.prototypes_)

        # Gaussian mixture probabilities
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        probs = jnp.exp(log_probs - log_norm)
        probs = probs / jnp.sum(probs, axis=1, keepdims=True)

        # Aggregate per class
        n_classes = self.n_classes_
        class_probs = jnp.zeros((X.shape[0], n_classes))
        for c in range(n_classes):
            mask = (self.prototype_labels_ == c).astype(jnp.float32)
            class_probs = class_probs.at[:, c].set(
                jnp.sum(probs * mask[None, :], axis=1)
            )
        return class_probs

    def transform(self, X):
        """Transform data into latent space."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _mlp_forward(self.backbone_params_, X, self.activation)

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'hidden_sizes': self.hidden_sizes,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'sigma': self.sigma,
            'loss_type': self.loss_type,
        })
        return hp

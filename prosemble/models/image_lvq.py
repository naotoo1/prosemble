"""
Image LVQ models: ImageGLVQ, ImageGMLVQ, ImageGTLVQ.

Siamese-style LVQ models with CNN backbones for image classification.
Both inputs and prototypes (stored as images) are passed through the
same CNN before distance computation.

Architecture::

    Image (H,W,C) ---> CNN ---> latent_x
                                    |
                                    v
    Proto (H,W,C) ---> CNN ---> latent_w    distance(latent_x, latent_w)
                                                       |
                                                       v
                                                   LVQ loss

References
----------
.. [1] Villmann, T., et al. (2017). Prototype-based Neural Network
       Layers: Incorporating Vector Quantization. arXiv:1812.01214.
"""

import jax
import jax.numpy as jnp

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import glvq_loss_with_transfer
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init, random_omega_init
from prosemble.models.lvqmln import _cnn_init, _cnn_forward
from prosemble.core.utils import orthogonalize


class ImageGLVQ(SupervisedPrototypeModel):
    """Image GLVQ — GLVQ with a CNN embedding network.

    Both input images and prototype images are passed through the same
    CNN backbone before computing squared Euclidean distances.

    Parameters
    ----------
    input_shape : tuple
        (height, width, channels) of input images.
    channels : list of int
        CNN output channels per layer. e.g. [16, 32].
    kernel_sizes : list of int
        Kernel sizes per conv layer. e.g. [3, 3].
    latent_dim : int
        Embedding dimension.
    activation : str
        CNN activation: 'relu', 'sigmoid', 'tanh', etc.
    beta : float
        GLVQ transfer function parameter.
    bb_lr : float, optional
        Separate learning rate for the backbone network. If None,
        uses the same lr as prototypes. Default: None.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Training epochs.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, input_shape=(28, 28, 1), channels=None,
                 kernel_sizes=None, latent_dim=32,
                 activation='relu', beta=10.0, bb_lr=None,
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
        self.input_shape = input_shape
        self.channels = channels or [16, 32]
        self.kernel_sizes = kernel_sizes or [3, 3]
        self.latent_dim = latent_dim
        self.activation = activation
        self.beta = beta
        self.bb_lr = bb_lr
        self.backbone_params_ = None

        if bb_lr is not None:
            self._optimizer = self._build_multi_lr_optimizer(
                self._optimizer_spec, self.lr, bb_lr
            )

    def _build_multi_lr_optimizer(self, optimizer, proto_lr, bb_lr):
        """Build optimizer with separate learning rates for prototypes and backbone."""
        import optax
        if not isinstance(optimizer, str):
            return optimizer
        proto_opt = self._build_optimizer(optimizer, proto_lr)
        bb_opt = self._build_optimizer(optimizer, bb_lr)
        return optax.multi_transform(
            {'prototypes': proto_opt, 'backbone': bb_opt},
            param_labels=lambda params: {k: k for k in params},
        )

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        return params

    def _init_state(self, X, y, key):
        key1, key2 = jax.random.split(key)

        backbone_params = _cnn_init(
            key1, self.input_shape, self.channels,
            self.kernel_sizes, self.latent_dim, self.activation,
        )

        # Prototypes in image space
        X_flat = X.reshape(X.shape[0], -1)
        prototypes_flat, proto_labels = self._init_prototypes(
            X_flat, y, self.n_prototypes_per_class, key2
        )
        prototypes = prototypes_flat.reshape(-1, *self.input_shape)

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
        }
        opt_state = self._optimizer.init(params)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes.reshape(-1, int(jnp.prod(jnp.array(self.input_shape)))),
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        backbone = params['backbone']
        protos = params['prototypes']
        # Reshape if flat
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = protos.reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(backbone, X_img, self.activation)
        latent_w = _cnn_forward(backbone, proto_img, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, latent_w)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']
        # Store prototypes as images
        self.prototypes_ = params['prototypes']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = self.prototypes_.reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(self.backbone_params_, X_img, self.activation)
        latent_w = _cnn_forward(self.backbone_params_, proto_img, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, latent_w)
        return wtac(distances, self.prototype_labels_)

    def predict_proba(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = self.prototypes_.reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(self.backbone_params_, X_img, self.activation)
        latent_w = _cnn_forward(self.backbone_params_, proto_img, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, latent_w)
        from prosemble.core.pooling import stratified_min_pooling
        class_dists = stratified_min_pooling(
            distances, self.prototype_labels_, self.n_classes_
        )
        return jax.nn.softmax(-class_dists, axis=1)

    def transform(self, X):
        """Transform images to latent space."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        return _cnn_forward(self.backbone_params_, X_img, self.activation)

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'input_shape': list(self.input_shape),
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        return hp


class ImageGMLVQ(SupervisedPrototypeModel):
    """Image GMLVQ — GMLVQ with a CNN embedding network.

    Like ImageGLVQ but with a learned Omega matrix in latent space:
    d = ||Omega(CNN(x) - CNN(w))||^2.

    Parameters
    ----------
    input_shape : tuple
        (height, width, channels).
    channels : list of int
        CNN channels per layer.
    kernel_sizes : list of int
        Kernel sizes per layer.
    latent_dim : int
        CNN output dimension.
    omega_dim : int, optional
        Omega mapping dimension. If None, uses latent_dim.
    activation : str
        CNN activation function.
    beta : float
        GLVQ transfer function parameter.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Training epochs.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, input_shape=(28, 28, 1), channels=None,
                 kernel_sizes=None, latent_dim=32,
                 omega_dim=None, activation='relu', beta=10.0,
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
        self.input_shape = input_shape
        self.channels = channels or [16, 32]
        self.kernel_sizes = kernel_sizes or [3, 3]
        self.latent_dim = latent_dim
        self.omega_dim = omega_dim
        self.activation = activation
        self.beta = beta
        self.backbone_params_ = None
        self.omega_ = None

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        params['omega'] = self.omega_
        return params

    def _init_state(self, X, y, key):
        omega_dim = self.omega_dim or self.latent_dim
        key1, key2 = jax.random.split(key)

        backbone_params = _cnn_init(
            key1, self.input_shape, self.channels,
            self.kernel_sizes, self.latent_dim, self.activation,
        )

        X_flat = X.reshape(X.shape[0], -1)
        prototypes_flat, proto_labels = self._init_prototypes(
            X_flat, y, self.n_prototypes_per_class, key2
        )
        prototypes = prototypes_flat.reshape(-1, *self.input_shape)
        omega = identity_omega_init(self.latent_dim, omega_dim)

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
            'omega': omega,
        }
        opt_state = self._optimizer.init(params)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes.reshape(-1, int(jnp.prod(jnp.array(self.input_shape)))),
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        backbone = params['backbone']
        omega = params['omega']
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = params['prototypes'].reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(backbone, X_img, self.activation)
        latent_w = _cnn_forward(backbone, proto_img, self.activation)
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        projected = jnp.einsum('npd,dl->npl', diff, omega)
        distances = jnp.sum(projected ** 2, axis=2)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']
        self.omega_ = params['omega']
        self.prototypes_ = params['prototypes']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = self.prototypes_.reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(self.backbone_params_, X_img, self.activation)
        latent_w = _cnn_forward(self.backbone_params_, proto_img, self.activation)
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        projected = jnp.einsum('npd,dl->npl', diff, self.omega_)
        distances = jnp.sum(projected ** 2, axis=2)
        return wtac(distances, self.prototype_labels_)

    def transform(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _cnn_forward(self.backbone_params_, X.reshape(-1, *self.input_shape), self.activation)

    @property
    def lambda_matrix(self):
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_.T @ self.omega_

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'input_shape': list(self.input_shape),
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        if self.omega_dim is not None:
            hp['omega_dim'] = self.omega_dim
        return hp


class ImageGTLVQ(SupervisedPrototypeModel):
    """Image GTLVQ — GTLVQ with a CNN embedding network.

    Like ImageGLVQ but with per-prototype tangent subspace bases in
    latent space: d = ||P_k(CNN(x) - CNN(w_k))||^2.

    Parameters
    ----------
    input_shape : tuple
        (height, width, channels).
    channels : list of int
        CNN channels per layer.
    kernel_sizes : list of int
        Kernel sizes per conv layer.
    latent_dim : int
        CNN output dimension.
    subspace_dim : int
        Tangent subspace dimension per prototype.
    activation : str
        CNN activation function.
    beta : float
        GLVQ transfer function parameter.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Training epochs.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, input_shape=(28, 28, 1), channels=None,
                 kernel_sizes=None, latent_dim=32,
                 subspace_dim=2, activation='relu', beta=10.0,
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
        self.input_shape = input_shape
        self.channels = channels or [16, 32]
        self.kernel_sizes = kernel_sizes or [3, 3]
        self.latent_dim = latent_dim
        self.subspace_dim = subspace_dim
        self.activation = activation
        self.beta = beta
        self.backbone_params_ = None
        self.omegas_ = None

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        key1, key2, key3 = jax.random.split(key, 3)

        backbone_params = _cnn_init(
            key1, self.input_shape, self.channels,
            self.kernel_sizes, self.latent_dim, self.activation,
        )

        X_flat = X.reshape(X.shape[0], -1)
        prototypes_flat, proto_labels = self._init_prototypes(
            X_flat, y, self.n_prototypes_per_class, key2
        )
        prototypes = prototypes_flat.reshape(-1, *self.input_shape)
        n_protos = prototypes.shape[0]

        keys = jax.random.split(key3, n_protos)
        omegas = jnp.stack([
            random_omega_init(self.latent_dim, self.subspace_dim, k) for k in keys
        ])

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
            'omegas': omegas,
        }
        opt_state = self._optimizer.init(params)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes.reshape(-1, int(jnp.prod(jnp.array(self.input_shape)))),
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        backbone = params['backbone']
        omegas = params['omegas']
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = params['prototypes'].reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(backbone, X_img, self.activation)
        latent_w = _cnn_forward(backbone, proto_img, self.activation)
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        proj = jnp.einsum('npd,pds->nps', diff, omegas)
        recon = jnp.einsum('nps,pds->npd', proj, omegas)
        tang_diff = diff - recon
        distances = jnp.sum(tang_diff ** 2, axis=2)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _post_update(self, params):
        if 'omegas' not in params:
            return params
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']
        self.omegas_ = params['omegas']
        self.prototypes_ = params['prototypes']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        proto_img = self.prototypes_.reshape(-1, *self.input_shape)
        latent_x = _cnn_forward(self.backbone_params_, X_img, self.activation)
        latent_w = _cnn_forward(self.backbone_params_, proto_img, self.activation)
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        proj = jnp.einsum('npd,pds->nps', diff, self.omegas_)
        recon = jnp.einsum('nps,pds->npd', proj, self.omegas_)
        tang_diff = diff - recon
        distances = jnp.sum(tang_diff ** 2, axis=2)
        return wtac(distances, self.prototype_labels_)

    def transform(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _cnn_forward(self.backbone_params_, X.reshape(-1, *self.input_shape), self.activation)

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'input_shape': list(self.input_shape),
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'latent_dim': self.latent_dim,
            'subspace_dim': self.subspace_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        return hp

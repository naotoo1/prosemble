"""
LVQ Multi-Layer Network (LVQMLN).

An MLP backbone transforms input data into a latent space where
prototypes reside. The GLVQ loss backpropagates through both the
backbone and prototype parameters jointly.

Unlike SiameseGLVQ (which transforms both inputs AND prototypes),
LVQMLN only transforms inputs — prototypes live directly in the
latent space.

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation, 21(12), 3532-3561.
.. [2] Villmann, T., et al. (2017). Prototype-based Neural Network
       Layers: Incorporating Vector Quantization. arXiv:1812.01214.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import glvq_loss_with_transfer
from prosemble.core.distance import squared_euclidean_distance_matrix


def _mlp_init(key, layer_sizes, activation='sigmoid'):
    """Initialize MLP parameters (Xavier/Glorot uniform).

    Parameters
    ----------
    key : JAX PRNGKey
    layer_sizes : list of int
        e.g. [4, 10, 2] means input=4, hidden=10, latent=2.

    Returns
    -------
    params : list of (weight, bias) tuples
    """
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        w = jax.random.uniform(subkey, (fan_in, fan_out), minval=-limit, maxval=limit)
        b = jnp.zeros(fan_out)
        params.append((w, b))
    return params


def _mlp_forward(params, x, activation='sigmoid'):
    """Forward pass through MLP.

    Parameters
    ----------
    params : list of (weight, bias)
    x : array of shape (n, d_in)
    activation : str
        Activation function for all layers.

    Returns
    -------
    array of shape (n, d_out)
    """
    act_fn = _get_activation(activation)
    for w, b in params:
        x = act_fn(x @ w + b)
    return x


def _get_activation(name):
    """Return JAX activation function by name."""
    if name == 'sigmoid':
        return jax.nn.sigmoid
    elif name == 'relu':
        return jax.nn.relu
    elif name == 'tanh':
        return jnp.tanh
    elif name == 'leaky_relu':
        return jax.nn.leaky_relu
    elif name == 'selu':
        return jax.nn.selu
    else:
        raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# CNN backbone for image data
# ---------------------------------------------------------------------------

def _cnn_init(key, input_shape, channels, kernel_sizes, latent_dim, activation='relu'):
    """Initialize CNN backbone parameters.

    Architecture: Conv layers -> global average pool -> linear -> latent_dim.

    Parameters
    ----------
    key : JAX PRNGKey
    input_shape : tuple
        (height, width, channels) of input images.
    channels : list of int
        Output channels per conv layer. e.g. [16, 32].
    kernel_sizes : list of int
        Kernel size per conv layer. e.g. [3, 3].
    latent_dim : int
        Output dimension after pooling + linear.
    activation : str

    Returns
    -------
    params : dict with 'conv_layers' and 'linear'
    """
    h, w, c_in = input_shape
    conv_params = []
    for i, (c_out, k) in enumerate(zip(channels, kernel_sizes)):
        key, subkey = jax.random.split(key)
        fan_in = k * k * c_in
        fan_out = k * k * c_out
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        kernel = jax.random.uniform(subkey, (k, k, c_in, c_out),
                                    minval=-limit, maxval=limit)
        bias = jnp.zeros(c_out)
        conv_params.append((kernel, bias))
        c_in = c_out

    # Linear head: last_channels -> latent_dim
    key, subkey = jax.random.split(key)
    last_c = channels[-1] if channels else c_in
    limit = jnp.sqrt(6.0 / (last_c + latent_dim))
    w_linear = jax.random.uniform(subkey, (last_c, latent_dim),
                                  minval=-limit, maxval=limit)
    b_linear = jnp.zeros(latent_dim)

    return {
        'conv_layers': conv_params,
        'linear': (w_linear, b_linear),
    }


def _cnn_forward(params, x, activation='relu'):
    """Forward pass through CNN backbone.

    Parameters
    ----------
    params : dict with 'conv_layers' and 'linear'
    x : array of shape (n, h, w, c)
    activation : str

    Returns
    -------
    array of shape (n, latent_dim)
    """
    act_fn = _get_activation(activation)

    # Conv layers with same padding
    for kernel, bias in params['conv_layers']:
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        x = x + bias[None, None, None, :]
        x = act_fn(x)

    # Global average pooling: (n, h, w, c) -> (n, c)
    x = jnp.mean(x, axis=(1, 2))

    # Linear head
    w, b = params['linear']
    x = x @ w + b
    x = act_fn(x)
    return x


class LVQMLN(SupervisedPrototypeModel):
    """LVQ Multi-Layer Network.

    An MLP backbone maps inputs into a latent space. Prototypes reside
    directly in that latent space. The GLVQ loss trains both the backbone
    and the prototypes jointly via gradient descent.

    Architecture::

        Input (d) -> MLP -> Latent (latent_dim)
                              |
                              v
                    distance(latent_x, prototypes)
                              |
                              v
                          GLVQ loss

    Parameters
    ----------
    hidden_sizes : list of int
        Sizes of hidden layers. e.g. [10] for one hidden layer of 10 units.
    latent_dim : int
        Dimension of the latent/embedding space where prototypes live.
    activation : str
        Activation function: 'sigmoid', 'relu', 'tanh', 'leaky_relu', 'selu'.
    beta : float
        Transfer function parameter for GLVQ loss.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate for both backbone and prototypes.
    """

    def __init__(self, hidden_sizes=None, latent_dim=2,
                 activation='sigmoid', beta=10.0, bb_lr=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes or [10]
        self.latent_dim = latent_dim
        self.activation = activation
        self.beta = beta
        self.bb_lr = bb_lr
        self.backbone_params_ = None

        # Rebuild optimizer with per-parameter LRs if bb_lr is specified
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
        """Initialize backbone + prototypes in latent space."""
        n_features = X.shape[1]
        n_classes = int(jnp.max(y)) + 1

        # Build layer sizes: input -> hidden... -> latent
        layer_sizes = [n_features] + list(self.hidden_sizes) + [self.latent_dim]

        key1, key2, key3 = jax.random.split(key, 3)

        # Initialize backbone
        backbone_params = _mlp_init(key1, layer_sizes, self.activation)

        # Project data into latent space for prototype initialization
        latent_X = _mlp_forward(backbone_params, X, self.activation)

        # Initialize prototypes in latent space using projected data
        latent_y = y
        prototypes, proto_labels = self._init_prototypes(
            latent_X, latent_y, self.n_prototypes_per_class, key2
        )

        # Pack all trainable params
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
        """GLVQ loss in latent space."""
        # Transform input through backbone
        latent_x = _mlp_forward(params['backbone'], X, self.activation)
        # Compute distances in latent space
        distances = squared_euclidean_distance_matrix(latent_x, params['prototypes'])
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        """Store backbone params alongside prototypes."""
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']

    def predict(self, X):
        """Predict class labels.

        Transforms X through the backbone, then finds nearest prototype.
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, self.prototypes_)
        from prosemble.core.competitions import wtac
        return wtac(distances, self.prototype_labels_)

    def predict_proba(self, X):
        """Predict class probabilities."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, self.prototypes_)
        from prosemble.core.pooling import stratified_min_pooling
        class_dists = stratified_min_pooling(
            distances, self.prototype_labels_, self.n_classes_
        )
        return jax.nn.softmax(-class_dists, axis=1)

    def transform(self, X):
        """Transform data into latent space.

        Parameters
        ----------
        X : array of shape (n, d)

        Returns
        -------
        latent : array of shape (n, latent_dim)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _mlp_forward(self.backbone_params_, X, self.activation)

    def _check_fitted(self):
        """Check that model has been fitted."""
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'hidden_sizes': self.hidden_sizes,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        return hp

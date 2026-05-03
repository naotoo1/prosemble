"""
Prototype and parameter initializers for prototype-based learning.

These functions generate initial prototypes, labels, and transformation
matrices for supervised and unsupervised models.
"""

import jax
import jax.numpy as jnp


def stratified_selection_init(X, y, n_per_class, key):
    """Initialize prototypes by randomly selecting samples per class.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data.
    y : array of shape (n_samples,)
        Training labels.
    n_per_class : int, list, or dict
        Number of prototypes per class.
        - int: same count for all classes.
        - list: index i gives the count for class i, e.g. ``[2, 2, 1]``.
        - dict: maps class label to count, e.g. ``{0: 2, 1: 2, 2: 1}``.
    key : jax.random.PRNGKey
        Random key.

    Returns
    -------
    prototypes : array of shape (n_prototypes, n_features)
    prototype_labels : array of shape (n_prototypes,)
    """
    classes = jnp.unique(y)

    all_protos = []
    all_labels = []

    for c in classes:
        c_int = int(c)
        if isinstance(n_per_class, dict):
            n = n_per_class[c_int]
        elif isinstance(n_per_class, (list, tuple)):
            n = n_per_class[c_int]
        else:
            n = n_per_class
        key, subkey = jax.random.split(key)
        class_mask = (y == c)
        class_indices = jnp.where(class_mask, size=int(jnp.sum(class_mask)))[0]
        selected = jax.random.choice(subkey, class_indices, shape=(n,),
                                     replace=len(class_indices) < n)
        all_protos.append(X[selected])
        all_labels.append(jnp.full(n, c, dtype=y.dtype))

    prototypes = jnp.concatenate(all_protos, axis=0)
    prototype_labels = jnp.concatenate(all_labels, axis=0)
    return prototypes, prototype_labels


def stratified_mean_init(X, y):
    """Initialize prototypes at the mean of each class.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data.
    y : array of shape (n_samples,)
        Training labels.

    Returns
    -------
    prototypes : array of shape (n_classes, n_features)
    prototype_labels : array of shape (n_classes,)
    """
    classes = jnp.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    prototypes = jnp.zeros((n_classes, n_features))
    for i, c in enumerate(classes):
        class_mask = (y == c)
        class_data = X * class_mask[:, None]
        class_mean = jnp.sum(class_data, axis=0) / jnp.maximum(jnp.sum(class_mask), 1)
        prototypes = prototypes.at[i].set(class_mean)

    return prototypes, classes


def random_normal_init(n_prototypes, n_features, key, mean=0.0, std=1.0):
    """Initialize prototypes from a normal distribution.

    Parameters
    ----------
    n_prototypes : int
    n_features : int
    key : jax.random.PRNGKey
    mean : float
    std : float

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    return mean + std * jax.random.normal(key, shape=(n_prototypes, n_features))


def identity_omega_init(n_features, n_dims=None):
    """Initialize omega as an identity (or truncated identity) matrix.

    Parameters
    ----------
    n_features : int
        Input dimensionality.
    n_dims : int, optional
        Projection dimensionality. Defaults to n_features (square).

    Returns
    -------
    array of shape (n_features, n_dims)
    """
    if n_dims is None:
        n_dims = n_features
    omega = jnp.zeros((n_features, n_dims))
    min_dim = min(n_features, n_dims)
    omega = omega.at[:min_dim, :min_dim].set(jnp.eye(min_dim))
    return omega


def random_omega_init(n_features, n_dims, key):
    """Initialize omega as a random orthogonal matrix via QR decomposition.

    Parameters
    ----------
    n_features : int
        Input dimensionality.
    n_dims : int
        Projection dimensionality.
    key : jax.random.PRNGKey

    Returns
    -------
    array of shape (n_features, n_dims)
    """
    A = jax.random.normal(key, shape=(n_features, n_dims))
    Q, _ = jnp.linalg.qr(A)
    return Q[:, :n_dims]


def uniform_init(n_prototypes, n_features, key, low=0.0, high=1.0):
    """Initialize prototypes from a uniform distribution.

    Parameters
    ----------
    n_prototypes : int
    n_features : int
    key : jax.random.PRNGKey
    low : float
    high : float

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    return jax.random.uniform(key, shape=(n_prototypes, n_features),
                              minval=low, maxval=high)


def zeros_init(n_prototypes, n_features):
    """Initialize prototypes as zeros.

    Useful for checkpoint loading where shapes must be pre-allocated
    before restoring saved values.

    Parameters
    ----------
    n_prototypes : int
    n_features : int

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    return jnp.zeros((n_prototypes, n_features))


def ones_init(n_prototypes, n_features):
    """Initialize prototypes as ones.

    Parameters
    ----------
    n_prototypes : int
    n_features : int

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    return jnp.ones((n_prototypes, n_features))


def fill_value_init(n_prototypes, n_features, value=0.0):
    """Initialize prototypes filled with a constant value.

    Parameters
    ----------
    n_prototypes : int
    n_features : int
    value : float
        Fill value.

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    return jnp.full((n_prototypes, n_features), value)


def selection_init(X, n_prototypes, key):
    """Initialize prototypes by uniformly sampling from data (classless).

    Suitable for unsupervised models like Neural Gas, SOM, and
    fuzzy clustering where no labels are available.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data.
    n_prototypes : int
        Number of prototypes to select.
    key : jax.random.PRNGKey

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    indices = jax.random.choice(key, X.shape[0], shape=(n_prototypes,),
                                replace=X.shape[0] < n_prototypes)
    return X[indices]


def mean_init(X, n_prototypes):
    """Initialize all prototypes at the data mean (classless).

    Suitable for unsupervised models. All prototypes start at the
    global mean and diverge during training.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data.
    n_prototypes : int
        Number of prototypes.

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    data_mean = jnp.mean(X, axis=0)
    return jnp.tile(data_mean, (n_prototypes, 1))


def literal_init(values):
    """Initialize prototypes from literal values.

    Used for warm-starting from another model's prototypes or from
    user-provided values.

    Parameters
    ----------
    values : array-like of shape (n_prototypes, n_features)
        Literal prototype values.

    Returns
    -------
    array of shape (n_prototypes, n_features)
    """
    return jnp.asarray(values, dtype=jnp.float32)


def stratified_noise_init(X, y, n_per_class, key, noise_std=0.1):
    """Initialize prototypes by selecting samples per class and adding noise.

    Combines stratified selection with Gaussian noise injection for
    diverse initial prototypes.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_per_class : int, list, or dict
        Number of prototypes per class (same formats as stratified_selection_init).
    key : jax.random.PRNGKey
    noise_std : float
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    prototypes : array of shape (n_prototypes, n_features)
    prototype_labels : array of shape (n_prototypes,)
    """
    key1, key2 = jax.random.split(key)
    prototypes, labels = stratified_selection_init(X, y, n_per_class, key1)
    noise = noise_std * jax.random.normal(key2, shape=prototypes.shape)
    return prototypes + noise, labels


def pca_omega_init(X, n_dims):
    """Initialize omega using PCA directions from training data.

    The top-n_dims principal components become the columns of omega,
    providing a data-driven initialization for metric learning models.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data.
    n_dims : int
        Number of principal components (projection dimensionality).

    Returns
    -------
    omega : array of shape (n_features, n_dims)
    """
    X_centered = X - jnp.mean(X, axis=0)
    _, _, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    return Vt[:n_dims].T


def class_conditional_mean_init(X, y, n_per_class):
    """Initialize prototypes at class means, replicated per n_per_class.

    When n_per_class > 1, each class gets multiple prototypes all initialized
    at the class mean (they will diverge during training).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_per_class : int, list, or dict
        Number of prototypes per class.

    Returns
    -------
    prototypes : array of shape (n_prototypes, n_features)
    prototype_labels : array of shape (n_prototypes,)
    """
    classes = jnp.unique(y)
    all_protos = []
    all_labels = []

    for c in classes:
        c_int = int(c)
        if isinstance(n_per_class, dict):
            n = n_per_class[c_int]
        elif isinstance(n_per_class, (list, tuple)):
            n = n_per_class[c_int]
        else:
            n = n_per_class
        class_mask = (y == c)
        class_data = X * class_mask[:, None]
        class_mean = jnp.sum(class_data, axis=0) / jnp.maximum(jnp.sum(class_mask), 1)
        all_protos.append(jnp.tile(class_mean, (n, 1)))
        all_labels.append(jnp.full(n, c, dtype=y.dtype))

    return jnp.concatenate(all_protos, axis=0), jnp.concatenate(all_labels, axis=0)


def random_reasonings_init(n_components, n_classes, key):
    """Initialize CBC reasoning matrices randomly.

    Parameters
    ----------
    n_components : int
    n_classes : int
    key : jax.random.PRNGKey

    Returns
    -------
    reasonings : array of shape (n_components, n_classes, 2)
    """
    return 0.5 + 0.01 * jax.random.normal(key, (n_components, n_classes, 2))


def pure_positive_reasonings_init(n_components, n_classes, key=None):
    """Initialize CBC reasoning matrices with pure positive evidence.

    Each component maps to exactly one class with high positive evidence
    and low negative evidence.

    Parameters
    ----------
    n_components : int
    n_classes : int
    key : ignored (included for API compatibility)

    Returns
    -------
    reasonings : array of shape (n_components, n_classes, 2)
    """
    reasonings = jnp.zeros((n_components, n_classes, 2))
    for i in range(n_components):
        c = i % n_classes
        reasonings = reasonings.at[i, c, 0].set(1.0)  # positive evidence
        reasonings = reasonings.at[i, c, 1].set(0.0)  # no negative evidence
    return reasonings

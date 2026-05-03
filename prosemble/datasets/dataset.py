"""
JAX-compatible dataset module.

Provides dataset loaders that return JAX arrays instead of NumPy arrays,
optimized for use with JAX-based clustering models.
"""

from dataclasses import dataclass

try:
    import jax.numpy as jnp
    from jax import Array as JaxArray
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JaxArray = None

from sklearn.datasets import load_breast_cancer, load_iris, make_moons, make_blobs


@dataclass
class DATASET_JAX:
    """
    JAX-compatible dataset container.

    Attributes
    ----------
    input_data : jax.numpy.ndarray
        Feature matrix as JAX array
    labels : jax.numpy.ndarray
        Labels as JAX array
    """
    input_data: 'JaxArray'
    labels: 'JaxArray'

    def to_numpy(self):
        """Convert JAX arrays back to NumPy arrays."""
        import numpy as np
        return {
            'input_data': np.array(self.input_data),
            'labels': np.array(self.labels)
        }


def iris_dataset_jax(dtype=None) -> DATASET_JAX:
    """
    Load Iris dataset as JAX arrays.

    Parameters
    ----------
    dtype : jax.numpy.dtype, default=jnp.float32
        Data type for features

    Returns
    -------
    DATASET_JAX
        Dataset with JAX arrays (150 samples, 4 features, 3 classes)

    Examples
    --------
    >>> from prosemble.datasets import iris_dataset_jax
    >>> dataset = iris_dataset_jax()
    >>> dataset.input_data.shape
    (150, 4)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for JAX datasets. Install with: pip install jax")

    if dtype is None:
        dtype = jnp.float32

    data, labels = load_iris(return_X_y=True)
    return DATASET_JAX(
        input_data=jnp.array(data, dtype=dtype),
        labels=jnp.array(labels, dtype=jnp.int32)
    )


def breast_cancer_dataset_jax(dtype=None) -> DATASET_JAX:
    """
    Load Wisconsin Breast Cancer dataset as JAX arrays.

    Parameters
    ----------
    dtype : jax.numpy.dtype, default=jnp.float32
        Data type for features

    Returns
    -------
    DATASET_JAX
        Dataset with JAX arrays (569 samples, 30 features, 2 classes)

    Examples
    --------
    >>> from prosemble.datasets import breast_cancer_dataset_jax
    >>> dataset = breast_cancer_dataset_jax()
    >>> dataset.input_data.shape
    (569, 30)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for JAX datasets. Install with: pip install jax")

    if dtype is None:
        dtype = jnp.float32

    data, labels = load_breast_cancer(return_X_y=True)
    return DATASET_JAX(
        input_data=jnp.array(data, dtype=dtype),
        labels=jnp.array(labels, dtype=jnp.int32)
    )


def moons_dataset_jax(
    n_samples: int = 150,
    noise: float | None = None,
    random_state: int | None = None,
    dtype=None
) -> DATASET_JAX:
    """
    Generate two interleaving half circles (moons) as JAX arrays.

    Parameters
    ----------
    n_samples : int, default=150
        Total number of points generated
    noise : float, optional
        Standard deviation of Gaussian noise added to data
    random_state : int, optional
        Random seed for reproducibility
    dtype : jax.numpy.dtype, default=jnp.float32
        Data type for features

    Returns
    -------
    DATASET_JAX
        Dataset with JAX arrays (n_samples, 2 features, 2 classes)

    Examples
    --------
    >>> from prosemble.datasets import moons_dataset_jax
    >>> dataset = moons_dataset_jax(n_samples=200, noise=0.1, random_state=42)
    >>> dataset.input_data.shape
    (200, 2)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for JAX datasets. Install with: pip install jax")

    if dtype is None:
        dtype = jnp.float32

    data, labels = make_moons(
        n_samples=n_samples,
        shuffle=True,
        noise=noise,
        random_state=random_state
    )
    return DATASET_JAX(
        input_data=jnp.array(data, dtype=dtype),
        labels=jnp.array(labels, dtype=jnp.int32)
    )


def blobs_dataset_jax(
    n_samples: list = None,
    centers: list = None,
    cluster_std: list = None,
    random_state: int | None = None,
    dtype=None
) -> DATASET_JAX:
    """
    Generate isotropic Gaussian blobs as JAX arrays.

    Parameters
    ----------
    n_samples : list, default=[120, 80]
        Number of samples per cluster
    centers : list, default=[[0.0, 0.0], [2.0, 2.0]]
        Centers of the clusters
    cluster_std : list, default=[1.2, 0.5]
        Standard deviation of clusters
    random_state : int, optional
        Random seed for reproducibility
    dtype : jax.numpy.dtype, default=jnp.float32
        Data type for features

    Returns
    -------
    DATASET_JAX
        Dataset with JAX arrays (sum(n_samples), 2 features)

    Examples
    --------
    >>> from prosemble.datasets import blobs_dataset_jax
    >>> dataset = blobs_dataset_jax(random_state=42)
    >>> dataset.input_data.shape
    (200, 2)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for JAX datasets. Install with: pip install jax")

    if dtype is None:
        dtype = jnp.float32
    if n_samples is None:
        n_samples = [120, 80]
    if centers is None:
        centers = [[0.0, 0.0], [2.0, 2.0]]
    if cluster_std is None:
        cluster_std = [1.2, 0.5]

    data, labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        shuffle=False
    )
    return DATASET_JAX(
        input_data=jnp.array(data, dtype=dtype),
        labels=jnp.array(labels, dtype=jnp.int32)
    )


@dataclass
class DATA_JAX:
    """
    JAX dataset collection with convenient property access.

    Parameters
    ----------
    random : int, default=4
        Random seed for dataset generation
    sample_size : int, default=1000
        Default sample size (not currently used)
    dtype : jax.numpy.dtype, default=jnp.float32
        Data type for features

    Examples
    --------
    >>> from prosemble.datasets import DATA_JAX
    >>> data = DATA_JAX(random=42)
    >>> moons = data.S_1  # Moons dataset
    >>> blobs = data.S_2  # Blobs dataset
    >>> cancer = data.breast_cancer  # Breast cancer dataset
    """
    random: int = 4
    sample_size: int = 1000
    dtype = jnp.float32 if JAX_AVAILABLE else None

    @property
    def S_1(self) -> DATASET_JAX:
        """Moons dataset."""
        return moons_dataset_jax(
            n_samples=150,
            noise=None,
            random_state=self.random,
            dtype=self.dtype
        )

    @property
    def S_2(self) -> DATASET_JAX:
        """Blobs dataset."""
        return blobs_dataset_jax(
            random_state=self.random,
            dtype=self.dtype
        )

    @property
    def iris(self) -> DATASET_JAX:
        """Iris dataset."""
        return iris_dataset_jax(dtype=self.dtype)

    @property
    def breast_cancer(self) -> DATASET_JAX:
        """Breast cancer dataset."""
        return breast_cancer_dataset_jax(dtype=self.dtype)

    @property
    def moons(self) -> DATASET_JAX:
        """Alias for S_1 (moons dataset)."""
        return self.S_1

    @property
    def blobs(self) -> DATASET_JAX:
        """Alias for S_2 (blobs dataset)."""
        return self.S_2


# Convenience functions for quick dataset loading
def load_iris_jax(dtype=None) -> DATASET_JAX:
    """Quick loader for iris dataset."""
    return iris_dataset_jax(dtype=dtype)


def load_breast_cancer_jax(dtype=None) -> DATASET_JAX:
    """Quick loader for breast cancer dataset."""
    return breast_cancer_dataset_jax(dtype=dtype)


def load_moons_jax(n_samples: int = 150, noise: float | None = None,
                   random_state: int | None = None, dtype=None) -> DATASET_JAX:
    """Quick loader for moons dataset."""
    return moons_dataset_jax(n_samples, noise, random_state, dtype)


def load_blobs_jax(random_state: int | None = None, dtype=None) -> DATASET_JAX:
    """Quick loader for blobs dataset."""
    return blobs_dataset_jax(random_state=random_state, dtype=dtype)


# Backward-compatible alias
DATA = DATA_JAX

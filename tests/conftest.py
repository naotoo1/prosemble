"""
Shared pytest fixtures for all tests.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split


@pytest.fixture
def iris_data():
    """Load iris dataset."""
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def iris_train_test_split():
    """Load iris dataset and split into train/test."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def simple_2d_data():
    """Generate simple 2D clustered data."""
    X, y = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        random_state=42
    )
    return X, y


@pytest.fixture
def simple_3d_data():
    """Generate simple 3D clustered data."""
    X, y = make_blobs(
        n_samples=150,
        n_features=3,
        centers=3,
        cluster_std=0.5,
        random_state=42
    )
    return X, y


@pytest.fixture
def random_data():
    """Generate random data for edge case testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    return X, y

"""Tests for Classification-By-Components (CBC)."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.cbc import CBC


@pytest.fixture
def separable_2d():
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    return jnp.array(data.data, dtype=jnp.float32), jnp.array(data.target, dtype=jnp.int32)


class TestCBC:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = CBC(
            n_components=4, n_classes=2, sigma=0.5,
            max_iter=50, lr=0.05,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_predict_proba_shape(self, separable_2d):
        X, y = separable_2d
        model = CBC(
            n_components=4, n_classes=2, sigma=0.5,
            max_iter=30, lr=0.05,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 2)

    def test_components_and_reasonings(self, separable_2d):
        X, y = separable_2d
        model = CBC(
            n_components=3, n_classes=2, sigma=0.5,
            max_iter=20, lr=0.05,
        )
        model.fit(X, y)
        assert model.components_.shape == (3, 2)
        assert model.reasonings_.shape == (3, 2, 2)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = CBC(
            n_components=4, n_classes=2, sigma=0.5,
            max_iter=50, lr=0.05,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] <= model.loss_history_[0]

    def test_iris(self, iris_data):
        X, y = iris_data
        model = CBC(
            n_components=6, n_classes=3, sigma=2.0,
            max_iter=100, lr=0.01,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.5

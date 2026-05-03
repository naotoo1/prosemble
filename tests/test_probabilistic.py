"""Tests for probabilistic LVQ models."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.probabilistic_lvq import SLVQ, RSLVQ


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


class TestSLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = SLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = SLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_accuracy_iris(self, iris_data):
        X, y = iris_data
        model = SLVQ(sigma=2.0, n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.6

    def test_iris(self, iris_data):
        X, y = iris_data
        model = SLVQ(sigma=1.0, n_prototypes_per_class=1, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.6


class TestRSLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_accuracy(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=100, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

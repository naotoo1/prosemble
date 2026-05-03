"""Tests for non-gradient LVQ models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.lvq1 import LVQ1
from prosemble.models.lvq21 import LVQ21
from prosemble.models.median_lvq import MedianLVQ


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


class TestLVQ1:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = LVQ1(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_returns_self(self, separable_2d):
        X, y = separable_2d
        model = LVQ1(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        result = model.fit(X, y)
        assert result is model

    def test_prototypes_shape(self, separable_2d):
        X, y = separable_2d
        model = LVQ1(n_prototypes_per_class=2, max_iter=10, lr=0.1)
        model.fit(X, y)
        assert model.prototypes_.shape == (4, 2)

    def test_accuracy(self, separable_2d):
        X, y = separable_2d
        model = LVQ1(n_prototypes_per_class=1, max_iter=100, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = LVQ1(n_prototypes_per_class=1, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.6


class TestLVQ21:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = LVQ21(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_tracked(self, separable_2d):
        X, y = separable_2d
        model = LVQ21(n_prototypes_per_class=1, max_iter=20, lr=0.1)
        model.fit(X, y)
        assert model.loss_history_ is not None
        assert len(model.loss_history_) > 0

    def test_accuracy(self, separable_2d):
        X, y = separable_2d
        model = LVQ21(n_prototypes_per_class=1, max_iter=100, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75


class TestMedianLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = MedianLVQ(n_prototypes_per_class=1, max_iter=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_prototypes_are_data_points(self, separable_2d):
        X, y = separable_2d
        model = MedianLVQ(n_prototypes_per_class=1, max_iter=10)
        model.fit(X, y)
        # Each prototype should be an actual data point
        for i in range(model.prototypes_.shape[0]):
            proto = model.prototypes_[i]
            dists = jnp.sum((X - proto) ** 2, axis=1)
            assert jnp.min(dists) < 1e-8

"""Tests for GLVQ family models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.glvq import GLVQ, GLVQ1, GLVQ21


@pytest.fixture
def iris_subset():
    """Simple 3-class dataset (Iris-like)."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.array(data.target, dtype=jnp.int32)
    return X, y


@pytest.fixture
def separable_2d():
    """Easy 2-class 2D dataset."""
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestGLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.75

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)

        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_predict_proba(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (8, 2)
        np.testing.assert_allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_iris(self, iris_subset):
        X, y = iris_subset
        model = GLVQ(n_prototypes_per_class=1, max_iter=100, lr=0.01)
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.8

    def test_multiple_protos_per_class(self, iris_subset):
        X, y = iris_subset
        model = GLVQ(n_prototypes_per_class=3, max_iter=50, lr=0.01)
        model.fit(X, y)

        assert model.prototypes_.shape == (9, 4)  # 3 classes * 3 per class

    def test_with_sigmoid_transfer(self, separable_2d):
        from prosemble.core.activations import sigmoid_beta
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.1,
            transfer_fn=sigmoid_beta, beta=10.0,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None


class TestGLVQ1:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GLVQ1(n_prototypes_per_class=1, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)


class TestGLVQ21:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GLVQ21(n_prototypes_per_class=1, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = GLVQ21(n_prototypes_per_class=1, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

"""Tests for GRLVQ, GMLVQ, LGMLVQ, GTLVQ, CELVQ."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.relevance_lvq import GRLVQ
from prosemble.models.matrix_lvq import GMLVQ
from prosemble.models.local_matrix_lvq import LGMLVQ
from prosemble.models.tangent_lvq import GTLVQ
from prosemble.models.crossentropy_lvq import CELVQ


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


class TestGRLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GRLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_relevance_profile(self, separable_2d):
        X, y = separable_2d
        model = GRLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.05)
        model.fit(X, y)
        rel = model.relevance_profile
        assert rel.shape == (2,)
        np.testing.assert_allclose(jnp.sum(rel), 1.0, atol=1e-5)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = GRLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestGMLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GMLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_omega_matrix(self, separable_2d):
        X, y = separable_2d
        model = GMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.05)
        model.fit(X, y)
        omega = model.omega_matrix
        assert omega.shape == (2, 2)

    def test_lambda_matrix(self, separable_2d):
        X, y = separable_2d
        model = GMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.05)
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (2, 2)
        # Lambda should be symmetric PSD
        np.testing.assert_allclose(lam, lam.T, atol=1e-5)

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = GMLVQ(
            latent_dim=2, n_prototypes_per_class=1,
            max_iter=30, lr=0.01,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)

    def test_iris_accuracy(self, iris_data):
        X, y = iris_data
        model = GMLVQ(n_prototypes_per_class=1, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.8


class TestLGMLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = LGMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_local_omegas(self, separable_2d):
        X, y = separable_2d
        model = LGMLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.05)
        model.fit(X, y)
        assert model.omegas_.shape == (2, 2, 2)  # 2 protos, 2x2 omegas

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = LGMLVQ(
            latent_dim=2, n_prototypes_per_class=1,
            max_iter=20, lr=0.01,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)  # 3 protos, 4x2 omegas


class TestGTLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GTLVQ(
            subspace_dim=1, n_prototypes_per_class=1,
            max_iter=30, lr=0.05,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_orthogonality_maintained(self, separable_2d):
        X, y = separable_2d
        model = GTLVQ(
            subspace_dim=1, n_prototypes_per_class=1,
            max_iter=30, lr=0.05,
        )
        model.fit(X, y)
        for i in range(model.omegas_.shape[0]):
            omega_k = model.omegas_[i]  # (d, s)
            # Columns should have unit norm
            norms = jnp.linalg.norm(omega_k, axis=0)
            np.testing.assert_allclose(norms, 1.0, atol=1e-4)


class TestCELVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = CELVQ(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = CELVQ(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_iris(self, iris_data):
        X, y = iris_data
        model = CELVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.7

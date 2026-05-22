"""Tests for Differentiating Kernel models (Villmann, Haase & Kaden, 2015)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import (
    DKGLVQ, DKGRLVQ, DKGMLVQ,
    DKNeuralGas, DKKohonenSOM, DKHeskesSOM,
)
from prosemble.core.kernel import (
    kernel_distance_squared_per_proto,
    kernel_distance_squared_relevance,
    exponential_kernel_distance_squared,
)


# ─── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def separable_2d():
    """Easy 2-class 2D dataset."""
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def iris_data():
    """Iris dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.array(data.target, dtype=jnp.int32)
    return X, y


@pytest.fixture
def unsupervised_data():
    """Simple 2D data for unsupervised models."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X1 = jax.random.normal(k1, (30, 2)) + jnp.array([0.0, 0.0])
    X2 = jax.random.normal(k2, (30, 2)) + jnp.array([5.0, 5.0])
    return jnp.concatenate([X1, X2], axis=0)


# ─── Kernel Function Tests ────────────────────────────────────

class TestKernelFunctions:
    def test_per_proto_shape(self):
        X = jnp.ones((10, 4))
        W = jnp.zeros((3, 4))
        sigmas = jnp.array([1.0, 1.0, 1.0])
        D = kernel_distance_squared_per_proto(X, W, sigmas)
        assert D.shape == (10, 3)

    def test_per_proto_zero_distance(self):
        X = jnp.array([[1.0, 2.0]])
        W = jnp.array([[1.0, 2.0]])
        sigmas = jnp.array([1.0])
        D = kernel_distance_squared_per_proto(X, W, sigmas)
        np.testing.assert_allclose(D, 0.0, atol=1e-6)

    def test_per_proto_nonnegative(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (20, 5))
        W = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
        sigmas = jnp.array([0.5, 1.0, 1.5, 2.0])
        D = kernel_distance_squared_per_proto(X, W, sigmas)
        assert jnp.all(D >= 0)

    def test_per_proto_bounded(self):
        """Gaussian kernel distance is bounded: 0 <= d² <= 2."""
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (20, 5))
        W = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
        sigmas = jnp.ones(4)
        D = kernel_distance_squared_per_proto(X, W, sigmas)
        assert jnp.all(D <= 2.0 + 1e-6)

    def test_relevance_shape(self):
        X = jnp.ones((10, 4))
        W = jnp.zeros((3, 4))
        sigmas = jnp.array([1.0, 1.0, 1.0])
        relevances = jnp.array([0.25, 0.25, 0.25, 0.25])
        D = kernel_distance_squared_relevance(X, W, sigmas, relevances)
        assert D.shape == (10, 3)

    def test_exponential_kernel_shape(self):
        X = jnp.ones((10, 4))
        W = jnp.zeros((3, 4))
        omega_hat = 0.1 * jnp.eye(4, 3)
        D = exponential_kernel_distance_squared(X, W, omega_hat)
        assert D.shape == (10, 3)

    def test_exponential_kernel_self_distance_zero(self):
        X = jnp.array([[1.0, 2.0]])
        W = jnp.array([[1.0, 2.0]])
        omega_hat = 0.1 * jnp.eye(2)
        D = exponential_kernel_distance_squared(X, W, omega_hat)
        np.testing.assert_allclose(D, 0.0, atol=1e-5)

    def test_exponential_kernel_nonnegative(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (20, 5))
        W = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
        omega_hat = 0.1 * jnp.eye(5, 3)
        D = exponential_kernel_distance_squared(X, W, omega_hat)
        assert jnp.all(D >= -1e-5)


# ─── DKGLVQ Tests ─────────────────────────────────────────────

class TestDKGLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.75

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_sigmas_learned(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)
        assert model.sigmas_ is not None
        assert model.sigmas_.shape == (2,)
        assert jnp.all(model.sigmas_ >= model.sigma_min)

    def test_sigma_init_fixed(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(sigma_init=2.0, n_prototypes_per_class=1, max_iter=10, lr=0.1)
        model.fit(X, y)
        assert model.sigmas_ is not None

    def test_iris(self, iris_data):
        X, y = iris_data
        model = DKGLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.9

    def test_kernel_bandwidths_property(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (2,)

    def test_python_loop(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1, use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.75


# ─── DKGRLVQ Tests ────────────────────────────────────────────

class TestDKGRLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = DKGRLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.75

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = DKGRLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_relevances_learned(self, separable_2d):
        X, y = separable_2d
        model = DKGRLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)
        assert model.relevances_ is not None
        assert model.relevances_.shape == (2,)
        np.testing.assert_allclose(jnp.sum(model.relevances_), 1.0, atol=1e-5)

    def test_sigmas_learned(self, separable_2d):
        X, y = separable_2d
        model = DKGRLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.1)
        model.fit(X, y)
        assert model.sigmas_ is not None
        assert jnp.all(model.sigmas_ >= model.sigma_min)

    def test_iris(self, iris_data):
        X, y = iris_data
        model = DKGRLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.9


# ─── DKGMLVQ Tests ────────────────────────────────────────────

class TestDKGMLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ(n_prototypes_per_class=1, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.75

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_omega_hat_learned(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)
        model.fit(X, y)
        assert model.omega_hat_ is not None
        assert model.omega_hat_.shape == (2, 2)

    def test_lambda_hat_matrix(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)
        model.fit(X, y)
        L = model.lambda_hat_matrix
        assert L.shape == (2, 2)
        # Λ̂ = Ω̂ Ω̂^T should be symmetric PSD
        np.testing.assert_allclose(L, L.T, atol=1e-5)
        eigvals = jnp.linalg.eigvalsh(L)
        assert jnp.all(eigvals >= -1e-6)

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = DKGMLVQ(latent_dim=2, n_prototypes_per_class=1, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.omega_hat_.shape == (4, 2)
        L = model.lambda_hat_matrix
        assert L.shape == (4, 4)

    def test_numerical_stability(self, iris_data):
        """Verify no NaN/Inf with default settings."""
        X, y = iris_data
        model = DKGMLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)
        model.fit(X, y)
        assert jnp.all(jnp.isfinite(model.loss_history_))
        assert jnp.all(jnp.isfinite(model.omega_hat_))


# ─── DKNeuralGas Tests ────────────────────────────────────────

class TestDKNeuralGas:
    def test_fit(self, unsupervised_data):
        X = unsupervised_data
        model = DKNeuralGas(n_prototypes=5, kernel_sigma=2.0, max_iter=50)
        model.fit(X)
        assert model.prototypes_ is not None
        assert model.prototypes_.shape == (5, 2)

    def test_loss_decreases(self, unsupervised_data):
        X = unsupervised_data
        model = DKNeuralGas(n_prototypes=5, kernel_sigma=2.0, max_iter=50)
        model.fit(X)
        assert model.loss_history_[-1] <= model.loss_history_[0]

    def test_python_loop(self, unsupervised_data):
        X = unsupervised_data
        model = DKNeuralGas(n_prototypes=5, kernel_sigma=2.0, max_iter=30, use_scan=False)
        model.fit(X)
        assert model.prototypes_ is not None

    def test_predict(self, unsupervised_data):
        X = unsupervised_data
        model = DKNeuralGas(n_prototypes=5, kernel_sigma=2.0, max_iter=30)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (60,)


# ─── DKKohonenSOM Tests ──────────────────────────────────────

class TestDKKohonenSOM:
    def test_fit(self, unsupervised_data):
        X = unsupervised_data
        model = DKKohonenSOM(grid_height=3, grid_width=3, kernel_sigma=2.0, max_iter=30)
        model.fit(X)
        assert model.prototypes_ is not None
        assert model.prototypes_.shape == (9, 2)

    def test_bmu_map(self, unsupervised_data):
        X = unsupervised_data
        model = DKKohonenSOM(grid_height=3, grid_width=3, kernel_sigma=2.0, max_iter=30)
        model.fit(X)
        coords = model.bmu_map(X)
        assert coords.shape == (60, 2)

    def test_python_loop(self, unsupervised_data):
        X = unsupervised_data
        model = DKKohonenSOM(
            grid_height=3, grid_width=3, kernel_sigma=2.0,
            max_iter=20, use_scan=False,
        )
        model.fit(X)
        assert model.prototypes_ is not None


# ─── DKHeskesSOM Tests ────────────────────────────────────────

class TestDKHeskesSOM:
    def test_fit(self, unsupervised_data):
        X = unsupervised_data
        model = DKHeskesSOM(grid_height=3, grid_width=3, kernel_sigma=2.0, max_iter=30)
        model.fit(X)
        assert model.prototypes_ is not None
        assert model.prototypes_.shape == (9, 2)

    def test_bmu_map(self, unsupervised_data):
        X = unsupervised_data
        model = DKHeskesSOM(grid_height=3, grid_width=3, kernel_sigma=2.0, max_iter=30)
        model.fit(X)
        coords = model.bmu_map(X)
        assert coords.shape == (60, 2)

    def test_python_loop(self, unsupervised_data):
        X = unsupervised_data
        model = DKHeskesSOM(
            grid_height=3, grid_width=3, kernel_sigma=2.0,
            max_iter=20, use_scan=False,
        )
        model.fit(X)
        assert model.prototypes_ is not None

    def test_loss_decreases(self, unsupervised_data):
        X = unsupervised_data
        model = DKHeskesSOM(grid_height=3, grid_width=3, kernel_sigma=2.0, max_iter=50)
        model.fit(X)
        assert model.loss_history_[-1] <= model.loss_history_[0]


# ─── Gradient Flow Tests ──────────────────────────────────────

class TestGradientFlow:
    def test_dkglvq_gradients_finite(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ(n_prototypes_per_class=1, max_iter=3, lr=0.1)
        model.fit(X, y)
        # If we got here without NaN, gradients flowed correctly
        assert jnp.all(jnp.isfinite(model.prototypes_))
        assert jnp.all(jnp.isfinite(model.sigmas_))

    def test_dkgrlvq_gradients_finite(self, separable_2d):
        X, y = separable_2d
        model = DKGRLVQ(n_prototypes_per_class=1, max_iter=3, lr=0.1)
        model.fit(X, y)
        assert jnp.all(jnp.isfinite(model.prototypes_))
        assert jnp.all(jnp.isfinite(model.sigmas_))
        assert jnp.all(jnp.isfinite(model.relevances_))

    def test_dkgmlvq_gradients_finite(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ(n_prototypes_per_class=1, max_iter=3, lr=0.01)
        model.fit(X, y)
        assert jnp.all(jnp.isfinite(model.prototypes_))
        assert jnp.all(jnp.isfinite(model.omega_hat_))

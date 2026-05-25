"""Tests for Wasserstein LVQ models and distance functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.core.distance import (
    wasserstein2_distance_matrix,
    wasserstein2_omega_distance_matrix,
    wasserstein2_relevance_distance_matrix,
    squared_euclidean_distance_matrix,
)
from prosemble.models.wasserstein_glvq import WGLVQ
from prosemble.models.wasserstein_gmlvq import WGMLVQ
from prosemble.models.wasserstein_grlvq import WGRLVQ


# ---------------------------------------------------------------------------
# Distance function tests
# ---------------------------------------------------------------------------

class TestWasserstein2Distance:

    def test_analytical_correctness(self):
        """Hand-computed W2 distance."""
        # x = [1, 2], mu = [0, 0], sigma^2 = [1, 1]
        # W2^2 = (1-0)^2 + (2-0)^2 + 1 + 1 = 1 + 4 + 2 = 7
        X = jnp.array([[1.0, 2.0]])
        means = jnp.array([[0.0, 0.0]])
        log_vars = jnp.array([[0.0, 0.0]])  # exp(0) = 1
        D = wasserstein2_distance_matrix(X, means, log_vars)
        np.testing.assert_allclose(float(D[0, 0]), 7.0, atol=1e-5)

    def test_zero_variance_equals_euclidean(self):
        """When variance is zero, W2 should equal squared Euclidean."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (10, 4))
        means = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
        # log_var = -inf => var = 0, but use very small for numerical stability
        log_vars = jnp.full((3, 4), -30.0)  # exp(-30) ≈ 0
        w2 = wasserstein2_distance_matrix(X, means, log_vars)
        eucl = squared_euclidean_distance_matrix(X, means)
        np.testing.assert_allclose(np.asarray(w2), np.asarray(eucl), atol=1e-4)

    def test_variance_increases_distance(self):
        """Higher variance should increase W2 distance."""
        X = jnp.array([[0.0, 0.0]])
        means = jnp.array([[0.0, 0.0]])
        log_vars_small = jnp.array([[-2.0, -2.0]])
        log_vars_large = jnp.array([[2.0, 2.0]])
        d_small = wasserstein2_distance_matrix(X, means, log_vars_small)
        d_large = wasserstein2_distance_matrix(X, means, log_vars_large)
        assert float(d_large[0, 0]) > float(d_small[0, 0])

    def test_output_shape(self):
        """Output shape should be (n, p)."""
        X = jnp.ones((10, 5))
        means = jnp.ones((3, 5))
        log_vars = jnp.zeros((3, 5))
        D = wasserstein2_distance_matrix(X, means, log_vars)
        assert D.shape == (10, 3)

    def test_non_negative(self):
        """W2 distances must be non-negative."""
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (20, 4))
        means = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
        log_vars = jax.random.normal(jax.random.PRNGKey(2), (5, 4)) * 0.5
        D = wasserstein2_distance_matrix(X, means, log_vars)
        assert jnp.all(D >= 0.0)


class TestWasserstein2OmegaDistance:

    def test_identity_omega_matches_base(self):
        """With identity omega, should match base W2 distance."""
        X = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
        means = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
        log_vars = jnp.zeros((3, 4))
        omega = jnp.eye(4)
        d_omega = wasserstein2_omega_distance_matrix(X, means, log_vars, omega)
        d_base = wasserstein2_distance_matrix(X, means, log_vars)
        np.testing.assert_allclose(
            np.asarray(d_omega), np.asarray(d_base), atol=1e-4
        )

    def test_output_shape(self):
        """Output shape should be (n, p)."""
        X = jnp.ones((10, 5))
        means = jnp.ones((3, 5))
        log_vars = jnp.zeros((3, 5))
        omega = jnp.eye(5, 3)  # Project to 3D
        D = wasserstein2_omega_distance_matrix(X, means, log_vars, omega)
        assert D.shape == (10, 3)


class TestWasserstein2RelevanceDistance:

    def test_uniform_relevance_matches_scaled_euclidean(self):
        """With uniform relevances, should match scaled squared Euclidean + spread."""
        X = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
        means = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
        log_vars = jnp.zeros((3, 4))
        # Uniform relevances: softmax of equal values = 1/d each
        relevances = jnp.zeros(4)
        D = wasserstein2_relevance_distance_matrix(X, means, log_vars, relevances)
        assert D.shape == (10, 3)
        # Each component weighted by 0.25 + variance spread
        eucl = squared_euclidean_distance_matrix(X, means) * 0.25
        spread = jnp.sum(jnp.exp(log_vars), axis=1)
        expected = eucl + spread[None, :]
        np.testing.assert_allclose(np.asarray(D), np.asarray(expected), atol=1e-4)

    def test_output_shape(self):
        X = jnp.ones((10, 5))
        means = jnp.ones((3, 5))
        log_vars = jnp.zeros((3, 5))
        relevances = jnp.ones(5)
        D = wasserstein2_relevance_distance_matrix(X, means, log_vars, relevances)
        assert D.shape == (10, 3)


# ---------------------------------------------------------------------------
# WGLVQ model tests
# ---------------------------------------------------------------------------

def _make_iris_data():
    """Simple 3-class data for testing."""
    from prosemble.datasets import load_iris_jax
    dataset = load_iris_jax()
    return dataset.input_data, dataset.labels


class TestWGLVQ:

    def test_fit_loss_decreases(self):
        X, y = _make_iris_data()
        model = WGLVQ(
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    def test_predict_shape(self):
        X, y = _make_iris_data()
        model = WGLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_variance_positivity(self):
        X, y = _make_iris_data()
        model = WGLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert jnp.all(model.prototype_variances_ > 0.0)

    def test_prototype_means_match_prototypes(self):
        X, y = _make_iris_data()
        model = WGLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        np.testing.assert_array_equal(
            np.asarray(model.prototype_means_),
            np.asarray(model.prototypes_),
        )

    def test_save_load_roundtrip(self):
        X, y = _make_iris_data()
        model = WGLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = WGLVQ.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# WGMLVQ model tests
# ---------------------------------------------------------------------------

class TestWGMLVQ:

    def test_fit_loss_decreases(self):
        X, y = _make_iris_data()
        model = WGMLVQ(
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    def test_omega_shape(self):
        X, y = _make_iris_data()
        model = WGMLVQ(
            latent_dim=2, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)

    def test_lambda_matrix(self):
        X, y = _make_iris_data()
        model = WGMLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (4, 4)
        # Lambda should be symmetric
        np.testing.assert_allclose(
            np.asarray(lam), np.asarray(lam.T), atol=1e-5
        )

    def test_save_load_roundtrip(self):
        X, y = _make_iris_data()
        model = WGMLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = WGMLVQ.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# WGRLVQ model tests
# ---------------------------------------------------------------------------

class TestWGRLVQ:

    def test_fit_loss_decreases(self):
        X, y = _make_iris_data()
        model = WGRLVQ(
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    def test_relevance_profile_sums_to_one(self):
        X, y = _make_iris_data()
        model = WGRLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        rel = model.relevance_profile
        np.testing.assert_allclose(float(jnp.sum(rel)), 1.0, atol=1e-5)

    def test_relevance_non_negative(self):
        X, y = _make_iris_data()
        model = WGRLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert jnp.all(model.relevance_profile >= 0.0)

    def test_save_load_roundtrip(self):
        X, y = _make_iris_data()
        model = WGRLVQ(
            n_prototypes_per_class=1, max_iter=10,
            lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = WGRLVQ.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

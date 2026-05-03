"""Tests for similarity functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.similarities import (
    gaussian_similarity,
    cosine_similarity_matrix,
    euclidean_similarity,
)


class TestGaussianSimilarity:
    def test_zero_distance(self):
        """Zero distance -> similarity = 1."""
        result = gaussian_similarity(jnp.array([0.0]), variance=1.0)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_large_distance(self):
        """Large distance -> similarity near 0."""
        result = gaussian_similarity(jnp.array([100.0]), variance=1.0)
        assert result[0] < 1e-10

    def test_monotonically_decreasing(self):
        dists = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        result = gaussian_similarity(dists, variance=1.0)
        assert jnp.all(jnp.diff(result) < 0)

    def test_variance_effect(self):
        """Larger variance -> slower decay."""
        d = jnp.array([4.0])
        s_small = gaussian_similarity(d, variance=0.5)
        s_large = gaussian_similarity(d, variance=5.0)
        assert s_large > s_small

    def test_known_value(self):
        """exp(-4 / (2*1)) = exp(-2)."""
        result = gaussian_similarity(jnp.array([4.0]), variance=1.0)
        np.testing.assert_allclose(result, np.exp(-2.0), atol=1e-6)

    def test_shape_preserved(self):
        dists = jnp.ones((3, 4))
        result = gaussian_similarity(dists, variance=1.0)
        assert result.shape == (3, 4)

    def test_jit(self):
        import jax
        f = jax.jit(lambda d: gaussian_similarity(d, variance=2.0))
        result = f(jnp.array([1.0, 2.0]))
        assert result.shape == (2,)


class TestCosineSimilarityMatrix:
    def test_identical_vectors(self):
        X = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        result = cosine_similarity_matrix(X, X)
        np.testing.assert_allclose(jnp.diag(result), 1.0, atol=1e-5)

    def test_orthogonal_vectors(self):
        X = jnp.array([[1.0, 0.0]])
        Y = jnp.array([[0.0, 1.0]])
        result = cosine_similarity_matrix(X, Y)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-5)

    def test_opposite_vectors(self):
        X = jnp.array([[1.0, 0.0]])
        Y = jnp.array([[-1.0, 0.0]])
        result = cosine_similarity_matrix(X, Y)
        np.testing.assert_allclose(result[0, 0], -1.0, atol=1e-5)

    def test_shape(self):
        X = jnp.ones((5, 3))
        Y = jnp.ones((7, 3))
        result = cosine_similarity_matrix(X, Y)
        assert result.shape == (5, 7)

    def test_range(self):
        key = jax_key(42)
        import jax
        X = jax.random.normal(key, (10, 4))
        Y = jax.random.normal(jax.random.split(key)[0], (8, 4))
        result = cosine_similarity_matrix(X, Y)
        assert jnp.all(result >= -1.0 - 1e-5)
        assert jnp.all(result <= 1.0 + 1e-5)

    def test_scale_invariance(self):
        X = jnp.array([[1.0, 2.0]])
        Y = jnp.array([[3.0, 4.0]])
        s1 = cosine_similarity_matrix(X, Y)
        s2 = cosine_similarity_matrix(X * 5.0, Y * 0.1)
        np.testing.assert_allclose(s1, s2, atol=1e-5)


class TestEuclideanSimilarity:
    def test_zero_distance(self):
        X = jnp.array([[1.0, 2.0]])
        result = euclidean_similarity(X, X, variance=1.0)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-5)

    def test_shape(self):
        X = jnp.ones((4, 3))
        Y = jnp.ones((6, 3))
        result = euclidean_similarity(X, Y, variance=1.0)
        assert result.shape == (4, 6)

    def test_range(self):
        import jax
        key = jax.random.key(0)
        X = jax.random.normal(key, (5, 2))
        Y = jax.random.normal(jax.random.split(key)[0], (3, 2))
        result = euclidean_similarity(X, Y, variance=1.0)
        assert jnp.all(result > 0.0)
        assert jnp.all(result <= 1.0 + 1e-5)

    def test_symmetry(self):
        import jax
        key = jax.random.key(1)
        X = jax.random.normal(key, (3, 2))
        Y = jax.random.normal(jax.random.split(key)[0], (4, 2))
        s_xy = euclidean_similarity(X, Y, variance=1.0)
        s_yx = euclidean_similarity(Y, X, variance=1.0)
        np.testing.assert_allclose(s_xy, s_yx.T, atol=1e-5)


def jax_key(seed):
    import jax
    return jax.random.key(seed)

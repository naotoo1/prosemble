"""Tests for prototype and parameter initializers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.initializers import (
    stratified_selection_init,
    stratified_mean_init,
    random_normal_init,
    identity_omega_init,
    random_omega_init,
    uniform_init,
    zeros_init,
    ones_init,
    fill_value_init,
    selection_init,
    mean_init,
)


class TestStratifiedSelectionInit:
    def test_correct_count(self):
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = jnp.array([0, 0, 1, 1])
        key = jax.random.key(0)
        protos, labels = stratified_selection_init(X, y, n_per_class=1, key=key)
        assert protos.shape == (2, 2)  # 2 classes * 1 per class
        assert labels.shape == (2,)

    def test_labels_match_classes(self):
        X = jnp.ones((6, 3))
        y = jnp.array([0, 0, 1, 1, 2, 2])
        key = jax.random.key(1)
        protos, labels = stratified_selection_init(X, y, n_per_class=1, key=key)
        assert set(np.array(labels)) == {0, 1, 2}

    def test_multiple_per_class(self):
        X = jnp.ones((10, 2))
        y = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        key = jax.random.key(2)
        protos, labels = stratified_selection_init(X, y, n_per_class=3, key=key)
        assert protos.shape == (6, 2)  # 2 classes * 3 per class
        # 3 of each label
        assert jnp.sum(labels == 0) == 3
        assert jnp.sum(labels == 1) == 3

    def test_protos_from_data(self):
        """Selected prototypes should be actual data points."""
        X = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        y = jnp.array([0, 1, 0, 1])
        key = jax.random.key(3)
        protos, labels = stratified_selection_init(X, y, n_per_class=1, key=key)
        for p in protos:
            # Each prototype should match some data point
            dists = jnp.sum((X - p) ** 2, axis=1)
            assert jnp.min(dists) < 1e-10


class TestStratifiedMeanInit:
    def test_class_means(self):
        X = jnp.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
        y = jnp.array([0, 0, 1, 1])
        protos, labels = stratified_mean_init(X, y)
        assert protos.shape == (2, 2)
        # Class 0 mean: [2, 0], Class 1 mean: [0, 2]
        idx0 = int(jnp.where(labels == 0, size=1)[0][0])
        idx1 = int(jnp.where(labels == 1, size=1)[0][0])
        np.testing.assert_allclose(protos[idx0], [2.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(protos[idx1], [0.0, 2.0], atol=1e-5)

    def test_one_per_class(self):
        X = jnp.ones((20, 5))
        y = jnp.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
        protos, labels = stratified_mean_init(X, y)
        assert protos.shape == (4, 5)
        assert len(jnp.unique(labels)) == 4


class TestRandomNormalInit:
    def test_shape(self):
        key = jax.random.key(0)
        result = random_normal_init(5, 3, key)
        assert result.shape == (5, 3)

    def test_mean_std(self):
        key = jax.random.key(1)
        result = random_normal_init(10000, 1, key, mean=5.0, std=2.0)
        np.testing.assert_allclose(jnp.mean(result), 5.0, atol=0.1)
        np.testing.assert_allclose(jnp.std(result), 2.0, atol=0.1)


class TestIdentityOmegaInit:
    def test_square(self):
        result = identity_omega_init(4)
        np.testing.assert_allclose(result, jnp.eye(4), atol=1e-7)

    def test_rectangular(self):
        result = identity_omega_init(4, n_dims=2)
        assert result.shape == (4, 2)
        np.testing.assert_allclose(result, jnp.eye(4, 2), atol=1e-7)


class TestRandomOmegaInit:
    def test_shape(self):
        key = jax.random.key(0)
        result = random_omega_init(4, 2, key)
        assert result.shape == (4, 2)

    def test_orthogonal_columns(self):
        key = jax.random.key(1)
        Q = random_omega_init(5, 3, key)
        # Q^T Q should be close to identity
        np.testing.assert_allclose(Q.T @ Q, jnp.eye(3), atol=1e-5)

    def test_unit_norm_columns(self):
        key = jax.random.key(2)
        Q = random_omega_init(6, 4, key)
        norms = jnp.linalg.norm(Q, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestUniformInit:
    def test_shape(self):
        key = jax.random.key(0)
        result = uniform_init(5, 3, key)
        assert result.shape == (5, 3)

    def test_range(self):
        key = jax.random.key(1)
        result = uniform_init(1000, 2, key, low=-1.0, high=1.0)
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)


class TestOnesInit:
    def test_shape(self):
        result = ones_init(5, 3)
        assert result.shape == (5, 3)

    def test_values(self):
        result = ones_init(3, 4)
        np.testing.assert_allclose(result, jnp.ones((3, 4)))


class TestFillValueInit:
    def test_shape(self):
        result = fill_value_init(5, 3, value=7.0)
        assert result.shape == (5, 3)

    def test_values(self):
        result = fill_value_init(3, 4, value=-2.5)
        np.testing.assert_allclose(result, jnp.full((3, 4), -2.5))

    def test_default_zero(self):
        result = fill_value_init(2, 2)
        np.testing.assert_allclose(result, jnp.zeros((2, 2)))


class TestSelectionInit:
    def test_shape(self):
        X = jnp.ones((10, 4))
        key = jax.random.key(0)
        result = selection_init(X, 3, key)
        assert result.shape == (3, 4)

    def test_from_data(self):
        """Selected prototypes should be actual data points."""
        X = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        key = jax.random.key(1)
        result = selection_init(X, 2, key)
        for p in result:
            dists = jnp.sum((X - p) ** 2, axis=1)
            assert jnp.min(dists) < 1e-10

    def test_with_replacement(self):
        """When n_prototypes > n_samples, should sample with replacement."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        key = jax.random.key(2)
        result = selection_init(X, 5, key)
        assert result.shape == (5, 2)


class TestMeanInit:
    def test_shape(self):
        X = jnp.ones((10, 4))
        result = mean_init(X, 3)
        assert result.shape == (3, 4)

    def test_all_same(self):
        """All prototypes should be the data mean."""
        X = jnp.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
        result = mean_init(X, 3)
        expected_mean = jnp.array([1.0, 1.5])
        for i in range(3):
            np.testing.assert_allclose(result[i], expected_mean)

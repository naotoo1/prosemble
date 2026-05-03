"""Tests for stratified pooling operations."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.pooling import (
    stratified_min_pooling,
    stratified_sum_pooling,
    stratified_max_pooling,
    stratified_prod_pooling,
)


@pytest.fixture
def simple_setup():
    """3 prototypes (2 classes), 2 samples."""
    distances = jnp.array([
        [1.0, 5.0, 3.0],   # sample 0: d to proto 0,1,2
        [4.0, 2.0, 6.0],   # sample 1
    ])
    proto_labels = jnp.array([0, 1, 0])  # proto 0,2 -> class 0; proto 1 -> class 1
    return distances, proto_labels, 2


class TestStratifiedMinPooling:
    def test_known_values(self, simple_setup):
        distances, proto_labels, n_classes = simple_setup
        result = stratified_min_pooling(distances, proto_labels, n_classes)
        # sample 0: class 0 = min(1.0, 3.0)=1.0, class 1 = 5.0
        # sample 1: class 0 = min(4.0, 6.0)=4.0, class 1 = 2.0
        expected = jnp.array([[1.0, 5.0], [4.0, 2.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_shape(self, simple_setup):
        distances, proto_labels, n_classes = simple_setup
        result = stratified_min_pooling(distances, proto_labels, n_classes)
        assert result.shape == (2, 2)

    def test_single_class(self):
        distances = jnp.array([[1.0, 2.0, 3.0]])
        proto_labels = jnp.array([0, 0, 0])
        result = stratified_min_pooling(distances, proto_labels, 1)
        np.testing.assert_allclose(result, jnp.array([[1.0]]))

    def test_one_proto_per_class(self):
        distances = jnp.array([[3.0, 7.0]])
        proto_labels = jnp.array([0, 1])
        result = stratified_min_pooling(distances, proto_labels, 2)
        np.testing.assert_allclose(result, jnp.array([[3.0, 7.0]]))


class TestStratifiedSumPooling:
    def test_known_values(self, simple_setup):
        distances, proto_labels, n_classes = simple_setup
        result = stratified_sum_pooling(distances, proto_labels, n_classes)
        # sample 0: class 0 = 1.0+3.0=4.0, class 1 = 5.0
        # sample 1: class 0 = 4.0+6.0=10.0, class 1 = 2.0
        expected = jnp.array([[4.0, 5.0], [10.0, 2.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestStratifiedMaxPooling:
    def test_known_values(self, simple_setup):
        distances, proto_labels, n_classes = simple_setup
        result = stratified_max_pooling(distances, proto_labels, n_classes)
        # sample 0: class 0 = max(1.0, 3.0)=3.0, class 1 = 5.0
        expected = jnp.array([[3.0, 5.0], [6.0, 2.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestStratifiedProdPooling:
    def test_known_values(self, simple_setup):
        distances, proto_labels, n_classes = simple_setup
        result = stratified_prod_pooling(distances, proto_labels, n_classes)
        # sample 0: class 0 = 1.0*3.0=3.0, class 1 = 5.0
        expected = jnp.array([[3.0, 5.0], [24.0, 2.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-4)


class TestJIT:
    def test_min_pooling_jit(self, simple_setup):
        distances, proto_labels, n_classes = simple_setup
        # Just verify it runs twice (second call uses cached JIT)
        r1 = stratified_min_pooling(distances, proto_labels, n_classes)
        r2 = stratified_min_pooling(distances, proto_labels, n_classes)
        np.testing.assert_allclose(r1, r2)

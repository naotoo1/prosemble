"""Tests for competition mechanisms."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.competitions import wtac, knnc, cbcc


class TestWTAC:
    def test_known_values(self):
        distances = jnp.array([
            [1.0, 5.0],  # sample 0: closest to proto 0
            [5.0, 1.0],  # sample 1: closest to proto 1
        ])
        proto_labels = jnp.array([0, 1])
        result = wtac(distances, proto_labels)
        np.testing.assert_array_equal(result, jnp.array([0, 1]))

    def test_ties_broken_by_first(self):
        distances = jnp.array([[3.0, 3.0]])
        proto_labels = jnp.array([0, 1])
        result = wtac(distances, proto_labels)
        assert result[0] == 0  # argmin returns first index on tie

    def test_multiple_protos_per_class(self):
        distances = jnp.array([
            [5.0, 1.0, 3.0, 2.0],  # closest is proto 1 (label 0)
        ])
        proto_labels = jnp.array([0, 0, 1, 1])
        result = wtac(distances, proto_labels)
        assert result[0] == 0

    def test_shape(self):
        distances = jnp.ones((10, 5))
        proto_labels = jnp.array([0, 0, 1, 1, 2])
        result = wtac(distances, proto_labels)
        assert result.shape == (10,)


class TestKNNC:
    def test_k1_equals_wtac(self):
        distances = jnp.array([
            [1.0, 5.0, 3.0],
            [5.0, 1.0, 3.0],
        ])
        proto_labels = jnp.array([0, 1, 2])
        wtac_result = wtac(distances, proto_labels)
        knnc_result = knnc(distances, proto_labels, k=1, n_classes=3)
        np.testing.assert_array_equal(wtac_result, knnc_result)

    def test_majority_vote(self):
        # 3 nearest: labels [0, 0, 1] -> majority is 0
        distances = jnp.array([[1.0, 2.0, 3.0, 10.0]])
        proto_labels = jnp.array([0, 0, 1, 1])
        result = knnc(distances, proto_labels, k=3, n_classes=2)
        assert result[0] == 0

    def test_shape(self):
        distances = jnp.ones((5, 10))
        proto_labels = jnp.zeros(10, dtype=jnp.int32)
        result = knnc(distances, proto_labels, k=3, n_classes=1)
        assert result.shape == (5,)


class TestCBCC:
    def test_pure_positive_reasoning(self):
        """Each component reasons for exactly one class."""
        detections = jnp.array([[1.0, 0.0]])  # high detection for comp 0
        # comp 0 reasons for class 0, comp 1 reasons for class 1
        reasonings = jnp.array([
            [[1.0, 0.0], [0.0, 0.0]],  # comp 0: p=1 for class 0, p=0 for class 1
            [[0.0, 0.0], [1.0, 0.0]],  # comp 1: p=0 for class 0, p=1 for class 1
        ])
        result = cbcc(detections, reasonings)
        assert result.shape == (1, 2)
        # Class 0 should have higher probability
        assert result[0, 0] > result[0, 1]

    def test_shape(self):
        n_samples, n_components, n_classes = 5, 3, 4
        detections = jnp.ones((n_samples, n_components))
        reasonings = jnp.ones((n_components, n_classes, 2)) * 0.5
        result = cbcc(detections, reasonings)
        assert result.shape == (n_samples, n_classes)

    def test_output_bounded(self):
        detections = jnp.ones((3, 4))
        reasonings = jnp.ones((4, 2, 2)) * 0.5
        result = cbcc(detections, reasonings)
        assert jnp.all(jnp.isfinite(result))

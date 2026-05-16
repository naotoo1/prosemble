"""Tests for prosemble.core.data — batching and shuffling utilities."""

import pytest
import jax
import jax.numpy as jnp

from prosemble.core.data import shuffle_arrays, padded_batches, batched_iterator


class TestShuffleArrays:

    def test_same_permutation(self):
        key = jax.random.PRNGKey(0)
        X = jnp.arange(12).reshape(4, 3)
        y = jnp.array([10, 20, 30, 40])
        X_s, y_s = shuffle_arrays(key, X, y)

        # Each row of X_s should correspond to the same y_s label
        for i in range(4):
            # Find the original index of this row
            orig_idx = int(X_s[i, 0]) // 3
            assert y_s[i] == y[orig_idx]

    def test_shapes_preserved(self):
        key = jax.random.PRNGKey(1)
        a = jnp.ones((5, 3))
        b = jnp.ones((5,))
        c = jnp.ones((5, 2, 2))
        a_s, b_s, c_s = shuffle_arrays(key, a, b, c)
        assert a_s.shape == (5, 3)
        assert b_s.shape == (5,)
        assert c_s.shape == (5, 2, 2)

    def test_empty(self):
        key = jax.random.PRNGKey(2)
        result = shuffle_arrays(key)
        assert result == ()

    def test_single_array(self):
        key = jax.random.PRNGKey(3)
        X = jnp.arange(6).reshape(3, 2)
        X_s, = shuffle_arrays(key, X)
        assert X_s.shape == (3, 2)
        # All original rows should be present
        assert set(int(X_s[i, 0]) for i in range(3)) == {0, 2, 4}


class TestPaddedBatches:

    def test_exact_division(self):
        X = jnp.ones((8, 3))
        y = jnp.arange(8)
        X_b, y_b = padded_batches(X, y, batch_size=4)
        assert X_b.shape == (2, 4, 3)
        assert y_b.shape == (2, 4)

    def test_padding(self):
        X = jnp.ones((10, 3))
        y = jnp.arange(10)
        X_b, y_b = padded_batches(X, y, batch_size=4)
        # 10 -> 12 (3 batches of 4)
        assert X_b.shape == (3, 4, 3)
        assert y_b.shape == (3, 4)

    def test_no_labels(self):
        X = jnp.ones((10, 3))
        X_b, y_b = padded_batches(X, batch_size=4)
        assert X_b.shape == (3, 4, 3)
        assert y_b is None

    def test_with_shuffle(self):
        key = jax.random.PRNGKey(42)
        X = jnp.arange(20).reshape(10, 2)
        y = jnp.arange(10)
        X_b, y_b = padded_batches(X, y, batch_size=4, key=key)
        assert X_b.shape == (3, 4, 2)
        # After shuffling and padding, we should still have valid data
        assert not jnp.all(X_b[0] == X[:4])  # very likely shuffled

    def test_single_batch(self):
        X = jnp.ones((3, 2))
        X_b, y_b = padded_batches(X, batch_size=5)
        assert X_b.shape == (1, 5, 2)


class TestBatchedIterator:

    def test_basic_iteration(self):
        X = jnp.ones((10, 3))
        y = jnp.arange(10)
        batches = list(batched_iterator(X, y, batch_size=4, shuffle=False))
        assert len(batches) == 3
        assert batches[0][0].shape == (4, 3)
        assert batches[1][0].shape == (4, 3)
        assert batches[2][0].shape == (2, 3)  # last batch is smaller

    def test_drop_last(self):
        X = jnp.ones((10, 3))
        batches = list(batched_iterator(X, batch_size=4, shuffle=False, drop_last=True))
        assert len(batches) == 2
        for X_b, _ in batches:
            assert X_b.shape[0] == 4

    def test_no_labels(self):
        X = jnp.ones((10, 3))
        for X_b, y_b in batched_iterator(X, batch_size=4, shuffle=False):
            assert y_b is None

    def test_shuffle(self):
        key = jax.random.PRNGKey(0)
        X = jnp.arange(20).reshape(10, 2)
        y = jnp.arange(10)
        batches = list(batched_iterator(X, y, batch_size=4, key=key))
        # Check that data was shuffled
        first_batch_X = batches[0][0]
        assert not jnp.array_equal(first_batch_X, X[:4])

    def test_exact_division(self):
        X = jnp.ones((8, 3))
        batches = list(batched_iterator(X, batch_size=4, shuffle=False))
        assert len(batches) == 2
        for X_b, _ in batches:
            assert X_b.shape[0] == 4

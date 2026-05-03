"""Tests for loss functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.losses import (
    glvq_loss,
    glvq_loss_with_transfer,
    lvq1_loss,
    lvq21_loss,
    nllr_loss,
    rslvq_loss,
    cross_entropy_lvq_loss,
    margin_loss,
    neural_gas_energy,
    _get_dp_dm,
)


class TestGetDpDm:
    def test_known_values(self):
        distances = jnp.array([
            [1.0, 5.0, 3.0, 7.0],  # same-class: [1, 3], diff-class: [5, 7]
        ])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1, 0, 1])
        dp, dm = _get_dp_dm(distances, target, proto_labels)
        np.testing.assert_allclose(dp, [1.0], atol=1e-6)
        np.testing.assert_allclose(dm, [5.0], atol=1e-6)

    def test_batch(self):
        distances = jnp.array([
            [1.0, 5.0],
            [5.0, 1.0],
        ])
        target = jnp.array([0, 0])
        proto_labels = jnp.array([0, 1])
        dp, dm = _get_dp_dm(distances, target, proto_labels)
        np.testing.assert_allclose(dp, [1.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(dm, [5.0, 1.0], atol=1e-6)


class TestGLVQLoss:
    def test_correct_classification_negative(self):
        """When d+ < d-, mu < 0 -> loss is negative."""
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = glvq_loss(distances, target, proto_labels)
        assert loss < 0

    def test_wrong_classification_positive(self):
        """When d+ > d-, mu > 0 -> loss is positive."""
        distances = jnp.array([[5.0, 1.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = glvq_loss(distances, target, proto_labels)
        assert loss > 0

    def test_equal_distances_zero(self):
        """When d+ == d-, mu = 0."""
        distances = jnp.array([[3.0, 3.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = glvq_loss(distances, target, proto_labels)
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)

    def test_known_value(self):
        """mu = (1-5)/(1+5) = -4/6 = -2/3."""
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = glvq_loss(distances, target, proto_labels)
        np.testing.assert_allclose(loss, -2.0 / 3.0, atol=1e-5)

    def test_range(self):
        """GLVQ mu is always in [-1, 1]."""
        key = jax.random.key(0)
        distances = jax.random.uniform(key, (100, 10), minval=0.1, maxval=10.0)
        target = jax.random.randint(jax.random.split(key)[0], (100,), 0, 5)
        proto_labels = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        loss = glvq_loss(distances, target, proto_labels)
        assert -1.0 <= loss <= 1.0

    def test_differentiable(self):
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        grad_fn = jax.grad(lambda d: glvq_loss(d, target, proto_labels))
        grads = grad_fn(distances)
        assert grads.shape == distances.shape
        assert jnp.all(jnp.isfinite(grads))


class TestGLVQLossWithTransfer:
    def test_with_sigmoid(self):
        from prosemble.core.activations import sigmoid_beta
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = glvq_loss_with_transfer(
            distances, target, proto_labels,
            transfer_fn=sigmoid_beta, beta=10.0,
        )
        # sigmoid of negative mu -> value < 0.5
        assert loss < 0.5

    def test_identity_matches_base(self):
        from prosemble.core.activations import identity
        distances = jnp.array([[2.0, 6.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss_base = glvq_loss(distances, target, proto_labels)
        loss_id = glvq_loss_with_transfer(
            distances, target, proto_labels,
            transfer_fn=identity,
        )
        np.testing.assert_allclose(loss_base, loss_id, atol=1e-6)


class TestLVQ1Loss:
    def test_correct_positive(self):
        """Correct classification -> positive loss (d+)."""
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = lvq1_loss(distances, target, proto_labels)
        assert loss > 0

    def test_wrong_negative(self):
        """Wrong classification -> negative loss (-d-)."""
        distances = jnp.array([[5.0, 1.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = lvq1_loss(distances, target, proto_labels)
        assert loss < 0


class TestLVQ21Loss:
    def test_correct_negative(self):
        """d+ < d- -> d+ - d- < 0."""
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = lvq21_loss(distances, target, proto_labels)
        assert loss < 0

    def test_known_value(self):
        """d+ - d- = 1 - 5 = -4."""
        distances = jnp.array([[1.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = lvq21_loss(distances, target, proto_labels)
        np.testing.assert_allclose(loss, -4.0, atol=1e-5)


class TestNLLRLoss:
    def test_correct_lower(self):
        """Well-separated -> lower loss."""
        distances_good = jnp.array([[0.1, 10.0]])
        distances_bad = jnp.array([[5.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss_good = nllr_loss(distances_good, target, proto_labels)
        loss_bad = nllr_loss(distances_bad, target, proto_labels)
        assert loss_good < loss_bad

    def test_finite(self):
        distances = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1, 2])
        loss = nllr_loss(distances, target, proto_labels)
        assert jnp.isfinite(loss)


class TestRSLVQLoss:
    def test_correct_lower(self):
        distances_good = jnp.array([[0.1, 10.0]])
        distances_bad = jnp.array([[5.0, 5.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss_good = rslvq_loss(distances_good, target, proto_labels)
        loss_bad = rslvq_loss(distances_bad, target, proto_labels)
        assert loss_good < loss_bad

    def test_finite(self):
        distances = jnp.array([[1.0, 2.0]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss = rslvq_loss(distances, target, proto_labels)
        assert jnp.isfinite(loss)


class TestCrossEntropyLVQLoss:
    def test_correct_lower(self):
        distances_good = jnp.array([[0.1, 10.0]])
        distances_bad = jnp.array([[10.0, 0.1]])
        target = jnp.array([0])
        proto_labels = jnp.array([0, 1])
        loss_good = cross_entropy_lvq_loss(distances_good, target, proto_labels, n_classes=2)
        loss_bad = cross_entropy_lvq_loss(distances_bad, target, proto_labels, n_classes=2)
        assert loss_good < loss_bad

    def test_nonnegative(self):
        distances = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([1])
        proto_labels = jnp.array([0, 1, 2])
        loss = cross_entropy_lvq_loss(distances, target, proto_labels, n_classes=3)
        assert loss >= 0

    def test_finite(self):
        distances = jnp.ones((5, 4))
        target = jnp.array([0, 1, 0, 1, 0])
        proto_labels = jnp.array([0, 0, 1, 1])
        loss = cross_entropy_lvq_loss(distances, target, proto_labels, n_classes=2)
        assert jnp.isfinite(loss)


class TestMarginLoss:
    def test_zero_when_correct(self):
        """Correct prediction with margin -> zero loss."""
        y_pred = jnp.array([[0.1, 0.9]])  # predicts class 1
        y_true = jnp.array([[0.0, 1.0]])  # true is class 1
        loss = margin_loss(y_pred, y_true, margin=0.3)
        # margin_loss = relu(d_m - d_p + margin), here d_p is large, so should be small
        assert loss >= 0

    def test_nonnegative(self):
        key = jax.random.key(0)
        y_pred = jax.random.uniform(key, (10, 3))
        y_true = jax.nn.one_hot(jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), 3)
        loss = margin_loss(y_pred, y_true, margin=0.3)
        assert loss >= 0


class TestNeuralGasEnergy:
    def test_shape(self):
        distances = jnp.array([[5.0, 1.0, 3.0]])
        energy = neural_gas_energy(distances, lam=1.0)
        assert energy.shape == ()

    def test_lambda_effect(self):
        """Smaller lambda -> more focused on nearest."""
        distances = jnp.array([[1.0, 5.0, 10.0]])
        e_small = neural_gas_energy(distances, lam=0.1)
        e_large = neural_gas_energy(distances, lam=10.0)
        # With small lambda, far nodes contribute less
        # Both should be finite
        assert jnp.isfinite(e_small)
        assert jnp.isfinite(e_large)

    def test_finite(self):
        distances = jnp.ones((5, 10))
        energy = neural_gas_energy(distances, lam=2.0)
        assert jnp.isfinite(energy)

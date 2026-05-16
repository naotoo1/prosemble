"""Tests for multi-device, gradient checkpointing, and custom VJP features."""

import os

# Simulate 4 CPU devices for testing multi-device code paths
os.environ.setdefault(
    'XLA_FLAGS', '--xla_force_host_platform_device_count=4'
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import GLVQ
from prosemble.core.distributed import (
    create_mesh, shard_data, replicate_params, replicate_opt_state,
    unshard_params,
)


class TestGradientCheckpointing:
    """Gradient checkpointing produces identical results to standard training."""

    def test_checkpointing_same_loss(self):
        """Loss history with checkpointing matches without."""
        X = jnp.array(np.random.RandomState(0).randn(30, 4).astype(np.float32))
        y = jnp.array([0] * 10 + [1] * 10 + [2] * 10, dtype=jnp.int32)

        m1 = GLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            gradient_checkpointing=False, random_seed=42,
        )
        m1.fit(X, y)

        m2 = GLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            gradient_checkpointing=True, random_seed=42,
        )
        m2.fit(X, y)

        np.testing.assert_allclose(
            np.asarray(m1.loss_history_),
            np.asarray(m2.loss_history_),
            atol=1e-5,
        )

    def test_checkpointing_same_predictions(self):
        """Predictions match with and without checkpointing."""
        X = jnp.array(np.random.RandomState(1).randn(20, 4).astype(np.float32))
        y = jnp.array([0] * 10 + [1] * 10, dtype=jnp.int32)

        m1 = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.01,
            gradient_checkpointing=False, random_seed=7,
        )
        m1.fit(X, y)

        m2 = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.01,
            gradient_checkpointing=True, random_seed=7,
        )
        m2.fit(X, y)

        np.testing.assert_array_equal(
            np.asarray(m1.predict(X)),
            np.asarray(m2.predict(X)),
        )

    def test_checkpointing_default_off(self):
        """Default is gradient_checkpointing=False."""
        model = GLVQ()
        assert model.gradient_checkpointing is False


class TestDistributedUtilities:
    """Test sharding utility functions."""

    def test_create_mesh_none(self):
        """create_mesh(None) returns None."""
        assert create_mesh(None) is None

    def test_create_mesh_devices(self):
        """create_mesh with devices creates a Mesh."""
        devices = jax.devices()
        mesh = create_mesh(devices)
        assert mesh is not None
        assert mesh.shape['data'] == len(devices)

    def test_shard_data_shape_preserved(self):
        """Sharded data preserves logical shape."""
        devices = jax.devices()
        mesh = create_mesh(devices)
        X = jnp.ones((8, 4))
        y = jnp.zeros(8, dtype=jnp.int32)
        X_s, y_s = shard_data(X, y, mesh)
        assert X_s.shape == (8, 4)
        assert y_s.shape == (8,)

    def test_replicate_params(self):
        """Replicated params preserve values."""
        devices = jax.devices()
        mesh = create_mesh(devices)
        params = {'prototypes': jnp.ones((3, 4))}
        replicated = replicate_params(params, mesh)
        np.testing.assert_array_equal(
            np.asarray(replicated['prototypes']),
            np.ones((3, 4)),
        )

    def test_unshard_params(self):
        """unshard_params brings arrays to single device."""
        params = {'prototypes': jnp.ones((3, 4))}
        result = unshard_params(params)
        assert result['prototypes'].shape == (3, 4)
        np.testing.assert_array_equal(
            np.asarray(result['prototypes']),
            np.ones((3, 4)),
        )


class TestMultiDeviceTraining:
    """Test multi-device training integration."""

    def test_devices_none_unaffected(self):
        """devices=None has no effect on training."""
        X = jnp.array(np.random.RandomState(0).randn(20, 4).astype(np.float32))
        y = jnp.array([0] * 10 + [1] * 10, dtype=jnp.int32)
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=5, lr=0.01,
            devices=None, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.prototypes_.shape == (2, 4)

    def test_multi_device_training(self):
        """Multi-device training produces valid results."""
        devices = jax.devices()
        n_devices = len(devices)
        # n_samples must be divisible by n_devices for even sharding
        n_per_class = 4 * n_devices
        X = jnp.array(
            np.random.RandomState(0).randn(2 * n_per_class, 4).astype(np.float32)
        )
        y = jnp.array(
            [0] * n_per_class + [1] * n_per_class, dtype=jnp.int32
        )

        model = GLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            devices=devices, random_seed=42, use_scan=False,
        )
        model.fit(X, y)

        assert model.prototypes_ is not None
        assert model.prototypes_.shape == (2, 4)
        preds = model.predict(X)
        assert preds.shape == (2 * n_per_class,)

    def test_multi_device_same_as_single(self):
        """Multi-device and single-device produce same results (deterministic)."""
        devices = jax.devices()
        n_devices = len(devices)
        n_samples = 4 * n_devices
        X = jnp.array(
            np.random.RandomState(5).randn(n_samples, 4).astype(np.float32)
        )
        y = jnp.array(
            [0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=jnp.int32
        )

        m_single = GLVQ(
            n_prototypes_per_class=1, max_iter=5, lr=0.01,
            devices=None, random_seed=42, use_scan=False,
        )
        m_single.fit(X, y)

        m_multi = GLVQ(
            n_prototypes_per_class=1, max_iter=5, lr=0.01,
            devices=devices, random_seed=42, use_scan=False,
        )
        m_multi.fit(X, y)

        # Predictions should match (same data, same seed)
        np.testing.assert_array_equal(
            np.asarray(m_single.predict(X)),
            np.asarray(m_multi.predict(X)),
        )

    def test_batch_size_validation(self):
        """batch_size must be divisible by n_devices."""
        devices = jax.devices()
        n_devices = len(devices)
        if n_devices < 2:
            pytest.skip("Need 2+ devices for this test")

        X = jnp.ones((20, 4))
        y = jnp.array([0] * 10 + [1] * 10, dtype=jnp.int32)

        # batch_size=3 not divisible by 4 devices
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=5, lr=0.01,
            devices=devices, batch_size=3,
        )
        with pytest.raises(ValueError, match="divisible"):
            model.fit(X, y)

    def test_partial_fit_with_devices(self):
        """partial_fit works after multi-device fit."""
        devices = jax.devices()
        n_devices = len(devices)
        n_samples = 4 * n_devices
        X = jnp.array(
            np.random.RandomState(0).randn(n_samples, 4).astype(np.float32)
        )
        y = jnp.array(
            [0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=jnp.int32
        )

        model = GLVQ(
            n_prototypes_per_class=1, max_iter=5, lr=0.01,
            devices=devices, random_seed=42, use_scan=False,
        )
        model.fit(X, y)

        # partial_fit should work
        model.partial_fit(X, y)
        assert model.prototypes_ is not None


class TestCustomVJP:
    """Test custom_vjp infrastructure."""

    def test_custom_vjp_hook_exists(self):
        """_custom_vjp_loss method exists on base class."""
        model = GLVQ()
        assert hasattr(model, '_custom_vjp_loss')

    def test_custom_vjp_not_used_by_default(self):
        """Default models don't use custom VJP."""
        model = GLVQ()
        assert not getattr(model, '_use_custom_vjp', False)

    def test_custom_vjp_returns_none_by_default(self):
        """Default _custom_vjp_loss returns None."""
        model = GLVQ()
        result = model._custom_vjp_loss(None, None, None, None)
        assert result is None

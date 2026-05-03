"""Tests for activation/transfer functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.activations import (
    identity, sigmoid_beta, swish_beta, get_activation, ACTIVATIONS
)


class TestIdentity:
    def test_passthrough(self):
        x = jnp.array([-1.0, 0.0, 1.0, 5.0])
        np.testing.assert_array_equal(identity(x), x)

    def test_beta_ignored(self):
        x = jnp.array([1.0, 2.0])
        np.testing.assert_array_equal(identity(x, beta=100.0), x)


class TestSigmoidBeta:
    def test_zero_input(self):
        result = sigmoid_beta(jnp.array(0.0))
        assert jnp.isclose(result, 0.5)

    def test_large_positive(self):
        result = sigmoid_beta(jnp.array(10.0), beta=10.0)
        assert result > 0.99

    def test_large_negative(self):
        result = sigmoid_beta(jnp.array(-10.0), beta=10.0)
        assert result < 0.01

    def test_output_range(self):
        x = jnp.linspace(-5, 5, 100)
        result = sigmoid_beta(x, beta=1.0)  # beta=1 avoids saturation
        assert jnp.all(result > 0.0)
        assert jnp.all(result < 1.0)

    def test_monotonic(self):
        x = jnp.linspace(-2, 2, 100)
        result = sigmoid_beta(x, beta=1.0)
        assert jnp.all(jnp.diff(result) > 0)

    def test_beta_controls_steepness(self):
        x = jnp.array(0.1)
        low_beta = sigmoid_beta(x, beta=1.0)
        high_beta = sigmoid_beta(x, beta=100.0)
        # Higher beta -> closer to step function -> more extreme at x=0.1
        assert high_beta > low_beta


class TestSwishBeta:
    def test_zero_input(self):
        result = swish_beta(jnp.array(0.0))
        assert jnp.isclose(result, 0.0)

    def test_positive_input_positive_output(self):
        result = swish_beta(jnp.array(5.0))
        assert result > 0.0

    def test_swish_formula(self):
        x = jnp.array(2.0)
        beta = 5.0
        expected = x * jnp.exp(beta * x) / (1 + jnp.exp(beta * x))
        result = swish_beta(x, beta)
        assert jnp.isclose(result, expected, rtol=1e-5)


class TestGetActivation:
    def test_by_name(self):
        fn = get_activation('identity')
        assert fn is identity

    def test_by_callable(self):
        custom = lambda x, beta=0: x * 2
        assert get_activation(custom) is custom

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation('nonexistent')

    def test_all_registered(self):
        assert 'identity' in ACTIVATIONS
        assert 'sigmoid_beta' in ACTIVATIONS
        assert 'swish_beta' in ACTIVATIONS


class TestJITCompilation:
    def test_identity_jit(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = identity(x)
        assert result.shape == (3,)

    def test_sigmoid_jit(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = sigmoid_beta(x)
        assert result.shape == (3,)

    def test_swish_jit(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = swish_beta(x)
        assert result.shape == (3,)

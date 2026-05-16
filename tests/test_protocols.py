"""Tests for prosemble.core.protocols."""

import pytest
import jax
import jax.numpy as jnp

from prosemble.core.protocols import Manifold, CallbackLike
from prosemble.core.manifolds import SO, SPD, Grassmannian
from prosemble.core.callbacks import Callback


class TestManifoldProtocol:
    """Verify concrete manifolds satisfy the Manifold protocol."""

    def test_so_isinstance(self):
        assert isinstance(SO(3), Manifold)

    def test_spd_isinstance(self):
        assert isinstance(SPD(3), Manifold)

    def test_grassmannian_isinstance(self):
        assert isinstance(Grassmannian(4, 2), Manifold)

    def test_arbitrary_object_not_manifold(self):
        assert not isinstance("hello", Manifold)
        assert not isinstance(42, Manifold)
        assert not isinstance({}, Manifold)

    def test_partial_impl_not_manifold(self):
        """An object with only some methods should not satisfy the protocol."""
        class Partial:
            def distance(self, p, q):
                pass
        assert not isinstance(Partial(), Manifold)


class TestCallbackProtocol:
    """Verify Callback satisfies the CallbackLike protocol."""

    def test_callback_isinstance(self):
        assert isinstance(Callback(), CallbackLike)

    def test_custom_callback(self):
        class MyCallback:
            def on_fit_start(self, model, X):
                pass
            def on_iteration_end(self, model, info):
                pass
            def on_fit_end(self, model, info):
                pass
        assert isinstance(MyCallback(), CallbackLike)

    def test_incomplete_callback_not_match(self):
        class Incomplete:
            def on_fit_start(self, model, X):
                pass
        assert not isinstance(Incomplete(), CallbackLike)


class TestTypeAliases:
    """Smoke-test that type aliases are importable."""

    def test_imports(self):
        from prosemble.core.protocols import (
            DistanceMatrixFn,
            DistancePairwiseFn,
            SupervisedInitFn,
            UnsupervisedInitFn,
        )
        # They are just Callable aliases — no runtime check needed
        assert DistanceMatrixFn is not None
        assert DistancePairwiseFn is not None
        assert SupervisedInitFn is not None
        assert UnsupervisedInitFn is not None

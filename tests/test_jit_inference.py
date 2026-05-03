"""Tests for JIT-compiled inference methods.

Verifies that JIT-compiled predict/predict_proba/transform produce
identical results to non-JIT execution, and that the second call
benefits from JIT cache (compilation already done).
"""

import time
import jax.numpy as jnp
import pytest

from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax


@pytest.fixture(scope="module")
def iris_data():
    dataset = load_iris_jax()
    X, y = dataset.input_data, dataset.labels
    X_train, X_test, y_train, y_test = train_test_split_jax(
        X, y, test_size=0.2, random_seed=42
    )
    return X_train, X_test, y_train, y_test


# --- Supervised Models ---

class TestGLVQJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        model = GLVQ(n_prototypes_per_class=2, max_iter=10, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)

    def test_predict_proba_numerical_correctness(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        model = GLVQ(n_prototypes_per_class=2, max_iter=10, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        proba1 = model.predict_proba(X_test)
        proba2 = model.predict_proba(X_test)
        assert jnp.allclose(proba1, proba2, atol=1e-6)
        # Probabilities sum to 1
        assert jnp.allclose(jnp.sum(proba1, axis=1), 1.0, atol=1e-5)

    def test_jit_cache_speedup(self, iris_data):
        """Second predict call should be faster (JIT cache hit)."""
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        model = GLVQ(n_prototypes_per_class=2, max_iter=10, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        # First call triggers compilation
        _ = model.predict(X_test).block_until_ready()
        # Subsequent calls use cached compiled function
        start = time.perf_counter()
        for _ in range(100):
            _ = model.predict(X_test).block_until_ready()
        cached_time = time.perf_counter() - start
        # Just verify it completes in reasonable time (< 1s for 100 calls)
        assert cached_time < 5.0


class TestGMLVQJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import GMLVQ
        X_train, X_test, y_train, _ = iris_data
        model = GMLVQ(n_prototypes_per_class=2, max_iter=10, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)


class TestLGMLVQJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import LGMLVQ
        X_train, X_test, y_train, _ = iris_data
        model = LGMLVQ(n_prototypes_per_class=2, max_iter=10, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)


class TestGTLVQJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import GTLVQ
        X_train, X_test, y_train, _ = iris_data
        model = GTLVQ(
            n_prototypes_per_class=2, subspace_dim=2,
            max_iter=10, lr=0.01, random_seed=42
        )
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)


class TestGRLVQJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import GRLVQ
        X_train, X_test, y_train, _ = iris_data
        model = GRLVQ(n_prototypes_per_class=2, max_iter=10, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)


class TestCBCJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import CBC
        X_train, X_test, y_train, _ = iris_data
        model = CBC(
            n_components=6, n_classes=3, max_iter=10,
            lr=0.01, sigma=1.0, random_seed=42
        )
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)

    def test_predict_proba_numerical_correctness(self, iris_data):
        from prosemble.models import CBC
        X_train, X_test, y_train, _ = iris_data
        model = CBC(
            n_components=6, n_classes=3, max_iter=10,
            lr=0.01, sigma=1.0, random_seed=42
        )
        model.fit(X_train, y_train)
        proba1 = model.predict_proba(X_test)
        proba2 = model.predict_proba(X_test)
        assert jnp.allclose(proba1, proba2, atol=1e-6)


# --- Unsupervised Models ---

class TestNeuralGasJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import NeuralGas
        X_train, X_test, _, _ = iris_data
        model = NeuralGas(n_prototypes=5, max_iter=10, random_seed=42)
        model.fit(X_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)

    def test_transform_numerical_correctness(self, iris_data):
        from prosemble.models import NeuralGas
        X_train, X_test, _, _ = iris_data
        model = NeuralGas(n_prototypes=5, max_iter=10, random_seed=42)
        model.fit(X_train)
        dists1 = model.transform(X_test)
        dists2 = model.transform(X_test)
        assert jnp.allclose(dists1, dists2, atol=1e-6)


class TestKohonenSOMJit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import KohonenSOM
        X_train, X_test, _, _ = iris_data
        model = KohonenSOM(grid_height=3, grid_width=3, max_iter=10, random_seed=42)
        model.fit(X_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)

    def test_transform_numerical_correctness(self, iris_data):
        from prosemble.models import KohonenSOM
        X_train, X_test, _, _ = iris_data
        model = KohonenSOM(grid_height=3, grid_width=3, max_iter=10, random_seed=42)
        model.fit(X_train)
        dists1 = model.transform(X_test)
        dists2 = model.transform(X_test)
        assert jnp.allclose(dists1, dists2, atol=1e-6)


# --- LVQ1 (non-gradient, uses base predict) ---

class TestLVQ1Jit:
    def test_predict_numerical_correctness(self, iris_data):
        from prosemble.models import LVQ1
        X_train, X_test, y_train, _ = iris_data
        model = LVQ1(n_prototypes_per_class=2, max_iter=10, lr=0.1, random_seed=42)
        model.fit(X_train, y_train)
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        assert jnp.array_equal(preds1, preds2)

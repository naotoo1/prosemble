"""Tests for model quantization (float16, bfloat16, int8)."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import FCM, GLVQ, GMLVQ, GRLVQ, NeuralGas


@pytest.fixture
def fitted_fcm():
    key = jax.random.key(42)
    X = jax.random.normal(key, (100, 4))
    model = FCM(n_clusters=3, fuzzifier=2.0, max_iter=20)
    model.fit(X)
    return model, X


@pytest.fixture
def fitted_glvq():
    key = jax.random.key(42)
    X = jax.random.normal(key, (100, 4))
    y = jnp.concatenate([jnp.zeros(50), jnp.ones(50)]).astype(jnp.int32)
    model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.01)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_gmlvq():
    key = jax.random.key(42)
    X = jax.random.normal(key, (100, 4))
    y = jnp.concatenate([jnp.zeros(50), jnp.ones(50)]).astype(jnp.int32)
    model = GMLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.001, latent_dim=2)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_neural_gas():
    key = jax.random.key(42)
    X = jax.random.normal(key, (100, 4))
    model = NeuralGas(n_prototypes=5, max_iter=20)
    model.fit(X)
    return model, X


# --- Basic quantize/dequantize ---

class TestQuantizeFloat16:
    def test_fcm_quantize(self, fitted_fcm):
        model, X = fitted_fcm
        assert model.centroids_.dtype == jnp.float32
        model.quantize('float16')
        assert model.is_quantized
        assert model.quantized_dtype == 'float16'
        assert model.centroids_.dtype == jnp.float16

    def test_glvq_quantize(self, fitted_glvq):
        model, X, y = fitted_glvq
        model.quantize('float16')
        assert model.prototypes_.dtype == jnp.float16
        assert model.is_quantized

    def test_gmlvq_quantize(self, fitted_gmlvq):
        model, X, y = fitted_gmlvq
        model.quantize('float16')
        assert model.prototypes_.dtype == jnp.float16
        assert model.omega_.dtype == jnp.float16

    def test_neural_gas_quantize(self, fitted_neural_gas):
        model, X = fitted_neural_gas
        model.quantize('float16')
        assert model.prototypes_.dtype == jnp.float16

    def test_dequantize_restores_float32(self, fitted_glvq):
        model, X, y = fitted_glvq
        model.quantize('float16')
        model.dequantize()
        assert not model.is_quantized
        assert model.prototypes_.dtype == jnp.float32

    def test_dequantize_noop_on_float32(self, fitted_glvq):
        model, X, y = fitted_glvq
        model.dequantize()  # should be a no-op
        assert not model.is_quantized
        assert model.prototypes_.dtype == jnp.float32


class TestQuantizeBfloat16:
    def test_bfloat16(self, fitted_glvq):
        model, X, y = fitted_glvq
        model.quantize('bfloat16')
        assert model.quantized_dtype == 'bfloat16'
        assert model.prototypes_.dtype == jnp.bfloat16

    def test_dequantize_bfloat16(self, fitted_glvq):
        model, X, y = fitted_glvq
        original = model.prototypes_.copy()
        model.quantize('bfloat16')
        model.dequantize()
        assert model.prototypes_.dtype == jnp.float32
        # Values should be close (bfloat16 has less precision)
        np.testing.assert_allclose(
            np.asarray(model.prototypes_), np.asarray(original), atol=0.01
        )


class TestQuantizeInt8:
    def test_int8_quantize(self, fitted_glvq):
        model, X, y = fitted_glvq
        model.quantize('int8')
        assert model.quantized_dtype == 'int8'
        assert model.prototypes_.dtype == jnp.int8

    def test_int8_dequantize(self, fitted_glvq):
        model, X, y = fitted_glvq
        original = model.prototypes_.copy()
        model.quantize('int8')
        model.dequantize()
        assert model.prototypes_.dtype == jnp.float32
        # Int8 has limited precision — allow ~1% of range error
        abs_max = float(jnp.max(jnp.abs(original)))
        np.testing.assert_allclose(
            np.asarray(model.prototypes_), np.asarray(original),
            atol=abs_max / 127.0 + 1e-6,
        )

    def test_int8_gmlvq(self, fitted_gmlvq):
        model, X, y = fitted_gmlvq
        model.quantize('int8')
        assert model.prototypes_.dtype == jnp.int8
        assert model.omega_.dtype == jnp.int8
        model.dequantize()
        assert model.prototypes_.dtype == jnp.float32
        assert model.omega_.dtype == jnp.float32

    def test_int8_scales_stored(self, fitted_glvq):
        model, X, y = fitted_glvq
        model.quantize('int8')
        assert hasattr(model, '_int8_scales')
        assert 'prototypes_' in model._int8_scales


# --- Inference after quantization ---

class TestQuantizedInference:
    def test_fcm_predict_after_quantize(self, fitted_fcm):
        model, X = fitted_fcm
        preds_f32 = model.predict(X)
        model.quantize('float16')
        model.dequantize()  # need float32 for distance computation
        preds_after = model.predict(X)
        # Most predictions should match
        match_ratio = float(jnp.mean(preds_f32 == preds_after))
        assert match_ratio > 0.9

    def test_glvq_predict_after_quantize(self, fitted_glvq):
        model, X, y = fitted_glvq
        preds_f32 = model.predict(X)
        model.quantize('float16')
        model.dequantize()
        preds_after = model.predict(X)
        match_ratio = float(jnp.mean(preds_f32 == preds_after))
        assert match_ratio > 0.9


# --- Save/load with quantization ---

class TestQuantizedSaveLoad:
    def test_save_with_quantize_option(self, fitted_glvq):
        model, X, y = fitted_glvq
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_q16')
            model.save(path, quantize='float16')
            # Model in memory should still be float32
            assert model.prototypes_.dtype == jnp.float32
            assert not model.is_quantized

    def test_save_load_quantized(self, fitted_glvq):
        model, X, y = fitted_glvq
        preds_original = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_q16')
            # Quantize then save
            model.quantize('float16')
            model.save(path)

            loaded = GLVQ.load(path)
            assert loaded.is_quantized
            assert loaded.quantized_dtype == 'float16'
            assert loaded.prototypes_.dtype == jnp.float16

            # Dequantize and predict
            loaded.dequantize()
            preds_loaded = loaded.predict(X)
            match_ratio = float(jnp.mean(preds_original == preds_loaded))
            assert match_ratio > 0.9

    def test_save_load_int8(self, fitted_glvq):
        model, X, y = fitted_glvq

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_int8')
            model.quantize('int8')
            model.save(path)

            loaded = GLVQ.load(path)
            assert loaded.is_quantized
            assert loaded.quantized_dtype == 'int8'
            assert loaded.prototypes_.dtype == jnp.int8
            assert hasattr(loaded, '_int8_scales')

    def test_save_quantize_preserves_memory(self, fitted_fcm):
        """save(quantize=...) should not change in-memory model."""
        model, X = fitted_fcm
        original_centroids = model.centroids_.copy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model')
            model.save(path, quantize='float16')
        np.testing.assert_array_equal(
            np.asarray(model.centroids_), np.asarray(original_centroids)
        )
        assert not model.is_quantized


# --- Size reduction ---

class TestSizeReduction:
    def test_float16_smaller(self, fitted_glvq):
        model, X, y = fitted_glvq
        with tempfile.TemporaryDirectory() as tmpdir:
            path_f32 = os.path.join(tmpdir, 'f32')
            path_f16 = os.path.join(tmpdir, 'f16')
            model.save(path_f32)
            model.save(path_f16, quantize='float16')
            size_f32 = os.path.getsize(path_f32 + '.npz')
            size_f16 = os.path.getsize(path_f16 + '.npz')
            assert size_f16 < size_f32


# --- Validation ---

class TestQuantizeValidation:
    def test_invalid_dtype(self, fitted_glvq):
        model, X, y = fitted_glvq
        with pytest.raises(ValueError, match="dtype must be one of"):
            model.quantize('float64')

    def test_quantize_unfitted(self):
        model = GLVQ(n_prototypes_per_class=1)
        with pytest.raises(Exception):
            model.quantize('float16')

    def test_chained_calls(self, fitted_glvq):
        """quantize returns self for chaining."""
        model, X, y = fitted_glvq
        result = model.quantize('float16')
        assert result is model
        result = model.dequantize()
        assert result is model

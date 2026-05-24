"""Tests for Riemannian Differentiating Kernel models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.core.manifolds import SO, Grassmannian
from prosemble.models import (
    RiemannianDKGLVQ, RiemannianDKGRLVQ, RiemannianDKGMLVQ,
    RiemannianDKSMNG, RiemannianDKSLNG, RiemannianDKSTNG,
    RiemannianDKRSMNG, RiemannianDKMSMNG,
    RiemannianDKRSLNG, RiemannianDKMSLNG,
    RiemannianDKRSTNG, RiemannianDKMSTNG,
)


# ---------------------------------------------------------------------------
# Fixtures: generate labeled manifold data
# ---------------------------------------------------------------------------

def _make_so3_data(n_per_class=10, seed=0):
    """Two classes of SO(3) rotations."""
    manifold = SO(3)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_per_class * 2)
    points = jnp.stack([manifold.random_point(k) for k in keys])
    X_flat = points.reshape(n_per_class * 2, -1)
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    return manifold, X_flat, y


def _make_gr42_data(n_per_class=10, seed=0):
    """Two classes of Grassmannian(4,2) points."""
    manifold = Grassmannian(4, 2)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_per_class * 2)
    points = jnp.stack([manifold.random_point(k) for k in keys])
    X_flat = points.reshape(n_per_class * 2, -1)
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    return manifold, X_flat, y


MANIFOLD_DATA = [
    pytest.param(_make_so3_data, id='SO3'),
    pytest.param(_make_gr42_data, id='Gr42'),
]


# ---------------------------------------------------------------------------
# RiemannianDKGLVQ
# ---------------------------------------------------------------------------

class TestRiemannianDKGLVQ:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos_m = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            projected = manifold.project(protos_m[i])
            np.testing.assert_allclose(
                np.asarray(protos_m[i]), np.asarray(projected),
                atol=1e-4,
            )

    def test_gamma_decays(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_sigmas_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)  # 2 protos * 2 classes
        assert jnp.all(model.sigmas_ > 0)

    def test_sigma_init_fixed(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
            sigma_init=1.0,
        )
        model.fit(X, y)
        assert model.sigmas_ is not None

    def test_kernel_bandwidths_property(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (2,)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKGLVQ.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKGRLVQ
# ---------------------------------------------------------------------------

class TestRiemannianDKGRLVQ:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos_m = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            projected = manifold.project(protos_m[i])
            np.testing.assert_allclose(
                np.asarray(protos_m[i]), np.asarray(projected),
                atol=1e-4,
            )

    def test_gamma_decays(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_sigmas_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)

    def test_relevances_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        d_flat = X.shape[1]
        assert model.relevances_.shape == (d_flat,)

    def test_relevance_profile_sums_to_one(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        profile = model.relevance_profile
        np.testing.assert_allclose(float(jnp.sum(profile)), 1.0, atol=1e-5)

    def test_kernel_bandwidths_property(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (2,)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGRLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKGRLVQ.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKGMLVQ
# ---------------------------------------------------------------------------

class TestRiemannianDKGMLVQ:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos_m = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            projected = manifold.project(protos_m[i])
            np.testing.assert_allclose(
                np.asarray(protos_m[i]), np.asarray(projected),
                atol=1e-4,
            )

    def test_gamma_decays(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_omega_hat_learned(self):
        manifold, X, y = _make_so3_data()
        d_flat = X.shape[1]
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_hat_.shape == (d_flat, d_flat)

    def test_omega_hat_matrix_property(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        oh = model.omega_hat_matrix
        assert oh.shape[0] == X.shape[1]

    def test_lambda_hat_psd(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        lh = model.lambda_hat_matrix
        eigvals = jnp.linalg.eigvalsh(lh)
        assert jnp.all(eigvals >= -1e-6)

    def test_latent_dim(self):
        manifold, X, y = _make_so3_data()
        d_flat = X.shape[1]
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
            latent_dim=4,
        )
        model.fit(X, y)
        assert model.omega_hat_.shape == (d_flat, 4)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKGMLVQ(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKGMLVQ.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKSMNG
# ---------------------------------------------------------------------------

class TestRiemannianDKSMNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKSMNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_sigmas_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSMNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)

    def test_omega_and_sigmas_both_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_ is not None
        assert model.sigmas_ is not None

    def test_kernel_bandwidths_property(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (2,)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKSMNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKSLNG
# ---------------------------------------------------------------------------

class TestRiemannianDKSLNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_sigmas_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)

    def test_omegas_and_sigmas_both_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omegas_ is not None
        assert model.sigmas_ is not None

    def test_kernel_bandwidths_property(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (2,)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKSLNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKSTNG
# ---------------------------------------------------------------------------

class TestRiemannianDKSTNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKSTNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_sigmas_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSTNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)

    def test_omegas_and_sigmas_both_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omegas_ is not None
        assert model.sigmas_ is not None

    def test_kernel_bandwidths_property(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=5, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (2,)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKSTNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKRSMNG (Relevance kernel + SMNG base)
# ---------------------------------------------------------------------------

class TestRiemannianDKRSMNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKRSMNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKRSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_sigmas_and_relevances_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSMNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)
        assert model.relevances_ is not None

    def test_relevance_profile_sums_to_one(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        profile = model.relevance_profile
        np.testing.assert_allclose(float(jnp.sum(profile)), 1.0, atol=1e-5)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKRSMNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKMSMNG (Matrix kernel + SMNG base)
# ---------------------------------------------------------------------------

class TestRiemannianDKMSMNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKMSMNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKMSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_omega_hat_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_hat_ is not None
        # omega_hat operates in projected (latent) space
        proj_dim = model.omega_.shape[1]
        assert model.omega_hat_.shape == (proj_dim, proj_dim)

    def test_lambda_hat_psd(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        lh = model.lambda_hat_matrix
        eigvals = jnp.linalg.eigvalsh(lh)
        assert jnp.all(eigvals >= -1e-6)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKMSMNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKRSLNG (Relevance kernel + SLNG base)
# ---------------------------------------------------------------------------

class TestRiemannianDKRSLNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKRSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKRSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_sigmas_and_relevances_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)
        assert model.relevances_ is not None

    def test_relevance_profile_sums_to_one(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        profile = model.relevance_profile
        np.testing.assert_allclose(float(jnp.sum(profile)), 1.0, atol=1e-5)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKRSLNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKMSLNG (Matrix kernel + SLNG base)
# ---------------------------------------------------------------------------

class TestRiemannianDKMSLNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKMSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKMSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_omega_hat_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_hat_ is not None
        # omega_hat operates in projected (latent) space
        proj_dim = model.omegas_.shape[2]
        assert model.omega_hat_.shape == (proj_dim, proj_dim)

    def test_lambda_hat_psd(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        lh = model.lambda_hat_matrix
        eigvals = jnp.linalg.eigvalsh(lh)
        assert jnp.all(eigvals >= -1e-6)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKMSLNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKRSTNG (Relevance kernel + STNG base)
# ---------------------------------------------------------------------------

class TestRiemannianDKRSTNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKRSTNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKRSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_sigmas_and_relevances_learned(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSTNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)
        assert model.relevances_ is not None
        d_flat = X.shape[1]
        assert model.relevances_.shape == (d_flat,)

    def test_relevance_profile_sums_to_one(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        profile = model.relevance_profile
        np.testing.assert_allclose(float(jnp.sum(profile)), 1.0, atol=1e-5)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKRSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKRSTNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )


# ---------------------------------------------------------------------------
# RiemannianDKMSTNG (Matrix kernel + STNG base)
# ---------------------------------------------------------------------------

class TestRiemannianDKMSTNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKMSTNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianDKMSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.asarray(preds)).issubset({0, 1})

    def test_omega_hat_learned(self):
        manifold, X, y = _make_so3_data()
        d_flat = X.shape[1]
        model = RiemannianDKMSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_hat_ is not None
        # omega_hat operates on residual (d_flat space)
        assert model.omega_hat_.shape == (d_flat, d_flat)

    def test_lambda_hat_psd(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        lh = model.lambda_hat_matrix
        eigvals = jnp.linalg.eigvalsh(lh)
        assert jnp.all(eigvals >= -1e-6)

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianDKMSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            model.save(path)
            loaded = RiemannianDKMSTNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(
            np.asarray(preds_before), np.asarray(preds_after)
        )

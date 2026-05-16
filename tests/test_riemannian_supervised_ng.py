"""Tests for Supervised Riemannian Neural Gas variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.core.manifolds import SO, SPD, Grassmannian
from prosemble.models import (
    RiemannianSRNG, RiemannianSMNG, RiemannianSLNG, RiemannianSTNG,
)


# ---------------------------------------------------------------------------
# Fixtures: generate labeled manifold data
# ---------------------------------------------------------------------------

def _make_so3_data(n_per_class=10, seed=0):
    """Two classes of SO(3) rotations, separated by a known rotation."""
    manifold = SO(3)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_per_class * 2)
    points = jnp.stack([manifold.random_point(k) for k in keys])
    X_flat = points.reshape(n_per_class * 2, -1)
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    return manifold, X_flat, y


def _make_spd3_data(n_per_class=10, seed=0):
    """Two classes of SPD(3) matrices."""
    manifold = SPD(3)
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
    pytest.param(_make_spd3_data, id='SPD3'),
    pytest.param(_make_gr42_data, id='Gr42'),
]


# ---------------------------------------------------------------------------
# RiemannianSRNG
# ---------------------------------------------------------------------------

class TestRiemannianSRNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            assert manifold.belongs(protos[i]), f"Prototype {i} not on manifold"

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        unique_preds = set(np.asarray(preds).tolist())
        assert unique_preds.issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_save_load_roundtrip(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = RiemannianSRNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_gamma_decays(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual

    def test_scan_mode(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=True,
        )
        model.fit(X, y)
        assert model.loss_ is not None


# ---------------------------------------------------------------------------
# RiemannianSMNG
# ---------------------------------------------------------------------------

class TestRiemannianSMNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            assert manifold.belongs(protos[i]), f"Prototype {i} not on manifold"

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        unique_preds = set(np.asarray(preds).tolist())
        assert unique_preds.issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_omega_learned(self, make_data):
        manifold, X, y = make_data()
        d_flat = X.shape[1]
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=15, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_ is not None
        assert model.omega_.shape[0] == d_flat

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_save_load_roundtrip(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = RiemannianSMNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_latent_dim(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianSMNG(
            manifold=manifold, latent_dim=4,
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            use_scan=False,
        )
        model.fit(X, y)
        assert model.omega_.shape == (9, 4)

    def test_relevance_matrix(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        lam = model.relevance_matrix()
        assert lam.shape == (9, 9)
        # Lambda should be symmetric positive semi-definite
        np.testing.assert_allclose(lam, lam.T, atol=1e-5)


# ---------------------------------------------------------------------------
# RiemannianSLNG
# ---------------------------------------------------------------------------

class TestRiemannianSLNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            assert manifold.belongs(protos[i]), f"Prototype {i} not on manifold"

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        unique_preds = set(np.asarray(preds).tolist())
        assert unique_preds.issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_per_prototype_omegas(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSLNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        d_flat = X.shape[1]
        assert model.omegas_.shape[0] == n_protos
        assert model.omegas_.shape[1] == d_flat

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_save_load_roundtrip(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSLNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = RiemannianSLNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


# ---------------------------------------------------------------------------
# RiemannianSTNG
# ---------------------------------------------------------------------------

class TestRiemannianSTNG:

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_fit_loss_decreases(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSTNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_prototypes_on_manifold(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            assert manifold.belongs(protos[i]), f"Prototype {i} not on manifold"

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_predict_returns_valid_labels(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        unique_preds = set(np.asarray(preds).tolist())
        assert unique_preds.issubset({0, 1})

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_omegas_orthonormal(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        for i in range(model.omegas_.shape[0]):
            OtO = model.omegas_[i].T @ model.omegas_[i]
            np.testing.assert_allclose(
                OtO, jnp.eye(OtO.shape[0]), atol=1e-4,
                err_msg=f"Omega {i} not orthonormal"
            )

    @pytest.mark.parametrize('make_data', MANIFOLD_DATA)
    def test_save_load_roundtrip(self, make_data):
        manifold, X, y = make_data()
        model = RiemannianSTNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = RiemannianSTNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_subspace_dim(self):
        manifold, X, y = _make_so3_data()
        model = RiemannianSTNG(
            manifold=manifold, subspace_dim=3,
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            use_scan=False,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (2, 9, 3)

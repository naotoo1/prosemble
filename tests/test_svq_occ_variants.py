"""Tests for SVQ-OCC metric adaptation variants (M, LM, T)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.svq_occ_m import SVQOCC_M
from prosemble.models.svq_occ_lm import SVQOCC_LM
from prosemble.models.svq_occ_t import SVQOCC_T


@pytest.fixture
def occ_2d():
    """Simple 2D one-class dataset."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X_target = jax.random.normal(k1, (40, 2)) * 0.3
    X_outlier = jax.random.normal(k2, (20, 2)) * 0.3 + 3.0
    X = jnp.concatenate([X_target, X_outlier])
    y = jnp.concatenate([jnp.zeros(40, dtype=jnp.int32),
                         jnp.ones(20, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def occ_4d():
    """4D dataset for testing metric adaptation."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_t, n_o = 60, 30
    X_t_rel = jax.random.normal(k1, (n_t, 2)) * 0.3
    X_o_rel = jax.random.normal(k2, (n_o, 2)) * 0.3 + 3.0
    X_t_noise = jax.random.normal(k3, (n_t, 2)) * 2.0
    X_o_noise = jax.random.normal(k4, (n_o, 2)) * 2.0
    X = jnp.concatenate([
        jnp.concatenate([X_t_rel, X_t_noise], axis=1),
        jnp.concatenate([X_o_rel, X_o_noise], axis=1),
    ])
    y = jnp.concatenate([jnp.zeros(n_t, dtype=jnp.int32),
                         jnp.ones(n_o, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def iris_binary():
    """Iris binary OCC dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.where(jnp.array(data.target) == 0, 0, 1).astype(jnp.int32)
    return X, y


# ============================================================
# SVQ-OCC-M (Global Matrix)
# ============================================================

class TestSVQOCCMBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = SVQOCC_M(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_recall = float(jnp.mean(preds[y == 0] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestSVQOCCMOmega:
    def test_omega_shape_default(self, occ_4d):
        X, y = occ_4d
        model = SVQOCC_M(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.omega_.shape == (4, 4)  # d × d when latent_dim=None

    def test_omega_shape_custom(self, occ_4d):
        X, y = occ_4d
        model = SVQOCC_M(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, latent_dim=2,
        )
        model.fit(X, y)
        assert model.omega_.shape == (4, 2)

    def test_omega_matrix_property(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        omega = model.omega_matrix
        np.testing.assert_array_equal(omega, model.omega_)

    def test_lambda_matrix(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        lam = model.lambda_matrix
        # Lambda should be symmetric positive semi-definite
        np.testing.assert_allclose(
            np.asarray(lam), np.asarray(lam.T), atol=1e-5
        )

    @pytest.mark.parametrize("cost_fn", ['contrastive', 'brier', 'cross_entropy'])
    def test_all_costs(self, occ_2d, cost_fn):
        X, y = occ_2d
        model = SVQOCC_M(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, cost_function=cost_fn,
        )
        model.fit(X, y)
        assert np.isfinite(model.loss_)


class TestSVQOCCMSaveLoad:
    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m_model")
            model.save(path)
            loaded = SVQOCC_M.load(path)
            np.testing.assert_allclose(
                np.asarray(model.omega_), np.asarray(loaded.omega_)
            )
            np.testing.assert_array_equal(
                model.predict(X), loaded.predict(X)
            )


class TestSVQOCCMResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.loss_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1


class TestSVQOCCMReject:
    def test_reject_inherited(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_M(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)


# ============================================================
# SVQ-OCC-LM (Local Matrix)
# ============================================================

class TestSVQOCCLMBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_LM(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = SVQOCC_LM(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_recall = float(jnp.mean(preds[y == 0] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_LM(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestSVQOCCLMOmegas:
    def test_omegas_shape_default(self, occ_4d):
        X, y = occ_4d
        model = SVQOCC_LM(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 4)

    def test_omegas_shape_custom(self, occ_4d):
        X, y = occ_4d
        model = SVQOCC_LM(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, latent_dim=2,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)

    @pytest.mark.parametrize("cost_fn", ['contrastive', 'brier', 'cross_entropy'])
    def test_all_costs(self, occ_2d, cost_fn):
        X, y = occ_2d
        model = SVQOCC_LM(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, cost_function=cost_fn,
        )
        model.fit(X, y)
        assert np.isfinite(model.loss_)


class TestSVQOCCLMSaveLoad:
    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_LM(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lm_model")
            model.save(path)
            loaded = SVQOCC_LM.load(path)
            np.testing.assert_allclose(
                np.asarray(model.omegas_), np.asarray(loaded.omegas_)
            )
            np.testing.assert_array_equal(
                model.predict(X), loaded.predict(X)
            )


class TestSVQOCCLMResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_LM(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.loss_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_LM(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1


# ============================================================
# SVQ-OCC-T (Tangent)
# ============================================================

class TestSVQOCCTBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            subspace_dim=1,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.60

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = SVQOCC_T(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_recall = float(jnp.mean(preds[y == 0] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            subspace_dim=1,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestSVQOCCTOmegas:
    def test_omegas_shape(self, occ_4d):
        X, y = occ_4d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, subspace_dim=2,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)

    def test_omegas_orthonormal(self, occ_4d):
        """After training, omegas should remain orthonormal."""
        X, y = occ_4d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, subspace_dim=2,
        )
        model.fit(X, y)
        for k in range(model.omegas_.shape[0]):
            omega_k = model.omegas_[k]  # (d, s)
            product = omega_k.T @ omega_k  # should be ~identity
            np.testing.assert_allclose(
                np.asarray(product),
                np.eye(model.subspace_dim),
                atol=1e-4,
            )

    def test_custom_subspace_dim(self, occ_4d):
        X, y = occ_4d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 1)

    @pytest.mark.parametrize("cost_fn", ['contrastive', 'brier', 'cross_entropy'])
    def test_all_costs(self, occ_2d, cost_fn):
        X, y = occ_2d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, cost_function=cost_fn, subspace_dim=1,
        )
        model.fit(X, y)
        assert np.isfinite(model.loss_)


class TestSVQOCCTSaveLoad:
    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            subspace_dim=1,
        )
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "t_model")
            model.save(path)
            loaded = SVQOCC_T.load(path)
            np.testing.assert_allclose(
                np.asarray(model.omegas_), np.asarray(loaded.omegas_)
            )
            np.testing.assert_array_equal(
                model.predict(X), loaded.predict(X)
            )


class TestSVQOCCTResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
            subspace_dim=1,
        )
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.loss_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_T(
            n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
            subspace_dim=1,
        )
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1

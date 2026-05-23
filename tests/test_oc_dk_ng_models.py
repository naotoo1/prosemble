"""Tests for One-Class Differentiating Kernel models with Neural Gas cooperation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models import OCDKGLVQ_NG, OCDKGRLVQ_NG, OCDKGMLVQ_NG


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
    """4D dataset with 2 relevant + 2 noise features."""
    key = jax.random.PRNGKey(123)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    X_target_rel = jax.random.normal(k1, (50, 2)) * 0.3
    X_outlier_rel = jax.random.normal(k2, (30, 2)) * 0.3 + 3.0
    X_target_noise = jax.random.normal(k3, (50, 2)) * 2.0
    X_outlier_noise = jax.random.normal(k4, (30, 2)) * 2.0
    X_target = jnp.concatenate([X_target_rel, X_target_noise], axis=1)
    X_outlier = jnp.concatenate([X_outlier_rel, X_outlier_noise], axis=1)
    X = jnp.concatenate([X_target, X_outlier])
    y = jnp.concatenate([jnp.zeros(50, dtype=jnp.int32),
                         jnp.ones(30, dtype=jnp.int32)])
    return X, y


# ============================================================================
# OCDKGLVQ_NG Tests
# ============================================================================

class TestOCDKGLVQ_NG:

    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0, use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=100, lr=0.01,
                            target_label=0, use_scan=False)
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) != float(history[0])

    def test_gamma_decay(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_gamma_bounded(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0, gamma_init=5.0, gamma_final=0.5)
        model.fit(X, y)
        assert model.gamma_ >= 0.5
        assert model.gamma_ < 5.0

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=30, lr=0.01,
                            target_label=0)
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_sigmas_learned(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0, use_scan=False)
        model.fit(X, y)
        assert model.sigmas_ is not None
        assert model.sigmas_.shape == (3,)
        assert jnp.all(model.sigmas_ > 0)

    def test_decision_function_range(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0)
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (60,)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=100, lr=0.01,
                            target_label=0, use_scan=False)
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_predict_with_reject(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)
        possible = {0, 1, -1}
        actual = set(int(p) for p in preds)
        assert actual.issubset(possible)

    def test_kernel_bandwidths_property(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=20, lr=0.01,
                            target_label=0)
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (3,)
        assert jnp.all(bw >= model.sigma_min)

    def test_scan_mode(self, occ_2d):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0, use_scan=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_save_load_roundtrip(self, occ_2d, tmp_path):
        X, y = occ_2d
        model = OCDKGLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                            target_label=0)
        model.fit(X, y)
        path = str(tmp_path / "ocdkglvq_ng.npz")
        model.save(path)
        loaded = OCDKGLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)


# ============================================================================
# OCDKGRLVQ_NG Tests
# ============================================================================

class TestOCDKGRLVQ_NG:

    def test_fit_predict(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80,)
        target_acc = float(jnp.mean(preds[:50] == 0))
        assert target_acc >= 0.50

    def test_loss_decreases(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=100, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) != float(history[0])

    def test_gamma_decay(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_sigmas_learned(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        assert model.sigmas_ is not None
        assert model.sigmas_.shape == (3,)
        assert jnp.all(model.sigmas_ > 0)

    def test_relevances_learned(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        assert model.relevances_ is not None
        assert model.relevances_.shape == (4,)

    def test_relevance_profile_property(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        profile = model.relevance_profile
        assert profile.shape == (4,)
        assert abs(float(jnp.sum(profile)) - 1.0) < 1e-5
        assert jnp.all(profile > 0)

    def test_decision_function_range(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (80,)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_target_higher_scores(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=100, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:50]))
        mean_outlier = float(jnp.mean(scores[50:]))
        assert mean_target > mean_outlier

    def test_predict_with_reject(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (80,)
        possible = {0, 1, -1}
        actual = set(int(p) for p in preds)
        assert actual.issubset(possible)

    def test_kernel_bandwidths_property(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=20, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (3,)
        assert jnp.all(bw >= model.sigma_min)

    def test_scan_mode(self, occ_4d):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80,)

    def test_save_load_roundtrip(self, occ_4d, tmp_path):
        X, y = occ_4d
        model = OCDKGRLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        path = str(tmp_path / "ocdkgrlvq_ng.npz")
        model.save(path)
        loaded = OCDKGRLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)


# ============================================================================
# OCDKGMLVQ_NG Tests
# ============================================================================

class TestOCDKGMLVQ_NG:

    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.50

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=100, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) != float(history[0])

    def test_gamma_decay(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=30, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_omega_hat_learned(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        assert model.omega_hat_ is not None
        assert model.omega_hat_.shape == (2, 2)

    def test_omega_hat_matrix_property(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=20, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        omega = model.omega_hat_matrix
        assert omega.shape == (2, 2)

    def test_lambda_hat_matrix(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=20, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        lam = model.lambda_hat_matrix
        assert lam.shape == (2, 2)
        eigenvalues = jnp.linalg.eigvalsh(lam)
        assert jnp.all(eigenvalues >= -1e-6)

    def test_latent_dim(self, occ_4d):
        X, y = occ_4d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=20, lr=0.01,
                             target_label=0, latent_dim=2)
        model.fit(X, y)
        assert model.omega_hat_.shape == (4, 2)
        assert model.lambda_hat_matrix.shape == (4, 4)

    def test_decision_function_range(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (60,)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=100, lr=0.01,
                             target_label=0, use_scan=False)
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_predict_with_reject(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)
        possible = {0, 1, -1}
        actual = set(int(p) for p in preds)
        assert actual.issubset(possible)

    def test_scan_mode(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0, use_scan=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_numerical_stability(self, occ_2d):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=20, lr=0.01,
                             target_label=0, omega_hat_scale=0.01)
        model.fit(X, y)
        assert jnp.all(jnp.isfinite(model.prototypes_))
        assert jnp.all(jnp.isfinite(model.omega_hat_))
        assert jnp.all(jnp.isfinite(model.thetas_))

    def test_save_load_roundtrip(self, occ_2d, tmp_path):
        X, y = occ_2d
        model = OCDKGMLVQ_NG(n_prototypes=3, max_iter=50, lr=0.01,
                             target_label=0)
        model.fit(X, y)
        path = str(tmp_path / "ocdkgmlvq_ng.npz")
        model.save(path)
        loaded = OCDKGMLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)

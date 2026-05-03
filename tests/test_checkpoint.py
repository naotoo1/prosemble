"""Tests for checkpointing, warm starting, and resume training."""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import (
    FCM, PCM, PFCM, HCM, AFCM, FPCM, IPCM, IPCM2,
    KFCM, KPCM, KAFCM, KFPCM, KPFCM, KIPCM, KIPCM2,
    FuzzyClusteringBase,
)
from prosemble.models.glvq import GLVQ
from prosemble.models.matrix_lvq import GMLVQ
from prosemble.models.relevance_lvq import GRLVQ
from prosemble.models.base import NotFittedError


@pytest.fixture
def sample_data():
    """Simple 2D dataset for testing."""
    return jnp.array([
        [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
        [5.0, 8.0], [8.0, 8.0], [6.0, 7.5],
        [9.0, 2.0], [8.5, 1.5], [9.5, 2.5],
    ])


# ---- Save/Load round-trip tests ----

class TestSaveLoad:
    """Test save/load round-trip for representative models."""

    def _round_trip(self, model, X, tmp_path):
        """Helper: fit, save, load, compare."""
        model.fit(X)
        path = str(tmp_path / "model.npz")
        model.save(path)

        loaded = FuzzyClusteringBase.load(path)
        assert type(loaded).__name__ == type(model).__name__
        assert loaded.n_clusters == model.n_clusters
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.centroids_),
            np.asarray(model.centroids_),
        )
        if model.objective_history_ is not None:
            np.testing.assert_array_almost_equal(
                np.asarray(loaded.objective_history_),
                np.asarray(model.objective_history_),
            )
        assert loaded.n_iter_ == model.n_iter_
        return loaded

    def test_fcm_round_trip(self, sample_data, tmp_path):
        model = FCM(n_clusters=3, max_iter=20, random_seed=42)
        loaded = self._round_trip(model, sample_data, tmp_path)
        assert loaded.fuzzifier == model.fuzzifier
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.U_), np.asarray(model.U_)
        )
        # Predictions should match
        np.testing.assert_array_equal(
            np.asarray(loaded.predict(sample_data)),
            np.asarray(model.predict(sample_data)),
        )

    def test_pcm_round_trip(self, sample_data, tmp_path):
        model = PCM(n_clusters=3, max_iter=20, random_seed=42)
        loaded = self._round_trip(model, sample_data, tmp_path)
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.T_), np.asarray(model.T_)
        )
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.gamma_), np.asarray(model.gamma_)
        )

    def test_hcm_round_trip(self, sample_data, tmp_path):
        model = HCM(n_clusters=3, max_iter=20, random_seed=42)
        loaded = self._round_trip(model, sample_data, tmp_path)
        np.testing.assert_array_equal(
            np.asarray(loaded.labels_), np.asarray(model.labels_)
        )

    def test_pfcm_round_trip(self, sample_data, tmp_path):
        model = PFCM(n_clusters=3, max_iter=20, random_seed=42)
        loaded = self._round_trip(model, sample_data, tmp_path)
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.U_), np.asarray(model.U_)
        )
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.T_), np.asarray(model.T_)
        )
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.gamma_), np.asarray(model.gamma_)
        )

    def test_kfcm_round_trip(self, sample_data, tmp_path):
        model = KFCM(n_clusters=3, max_iter=20, random_seed=42)
        loaded = self._round_trip(model, sample_data, tmp_path)
        assert loaded.sigma == model.sigma
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.U_), np.asarray(model.U_)
        )

    def test_load_adds_npz_extension(self, sample_data, tmp_path):
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        model.fit(sample_data)
        path = str(tmp_path / "model")
        model.save(path)
        loaded = FuzzyClusteringBase.load(path)
        assert type(loaded).__name__ == 'FCM'


# ---- Warm start tests ----

class TestWarmStart:
    """Test fit with initial_centroids."""

    def test_fcm_warm_start(self, sample_data):
        """FCM with initial_centroids should converge."""
        centroids = sample_data[:3]  # First 3 points as centroids
        model = FCM(n_clusters=3, max_iter=50, random_seed=42)
        model.fit(sample_data, initial_centroids=centroids)

        assert model.centroids_ is not None
        assert model.U_ is not None
        labels = model.predict(sample_data)
        assert labels.shape == (9,)

    def test_hcm_warm_start(self, sample_data):
        model = HCM(n_clusters=3, max_iter=50, random_seed=42)
        centroids = sample_data[:3]
        model.fit(sample_data, initial_centroids=centroids)
        assert model.centroids_ is not None

    def test_pcm_warm_start(self, sample_data):
        model = PCM(n_clusters=3, max_iter=50, random_seed=42)
        centroids = sample_data[:3]
        model.fit(sample_data, initial_centroids=centroids)
        assert model.centroids_ is not None
        assert model.T_ is not None
        assert model.gamma_ is not None

    def test_kfcm_warm_start(self, sample_data):
        model = KFCM(n_clusters=3, max_iter=50, random_seed=42)
        centroids = sample_data[:3]
        model.fit(sample_data, initial_centroids=centroids)
        assert model.centroids_ is not None
        assert model.U_ is not None

    def test_wrong_shape_raises(self, sample_data):
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        wrong_centroids = jnp.ones((2, 2))  # Wrong n_clusters
        with pytest.raises(ValueError, match="does not match expected"):
            model.fit(sample_data, initial_centroids=wrong_centroids)


# ---- Resume training tests ----

class TestResume:
    """Test fit with resume=True."""

    def test_fcm_resume(self, sample_data):
        """Resume should not fail and should produce valid results."""
        model = FCM(n_clusters=3, max_iter=5, random_seed=42)
        model.fit(sample_data)
        obj_before = model.objective_

        # Resume with more iterations
        model.max_iter = 50
        model.fit(sample_data, resume=True)
        assert model.centroids_ is not None
        # Objective should be <= before (or very close)
        assert model.objective_ <= obj_before + 1e-6

    def test_hcm_resume(self, sample_data):
        model = HCM(n_clusters=3, max_iter=5, random_seed=42)
        model.fit(sample_data)
        model.max_iter = 50
        model.fit(sample_data, resume=True)
        assert model.centroids_ is not None

    def test_pcm_resume(self, sample_data):
        model = PCM(n_clusters=3, max_iter=5, random_seed=42)
        model.fit(sample_data)
        model.max_iter = 50
        model.fit(sample_data, resume=True)
        assert model.centroids_ is not None

    def test_kfcm_resume(self, sample_data):
        model = KFCM(n_clusters=3, max_iter=5, random_seed=42)
        model.fit(sample_data)
        model.max_iter = 50
        model.fit(sample_data, resume=True)
        assert model.centroids_ is not None

    def test_resume_unfitted_raises(self, sample_data):
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        with pytest.raises(NotFittedError):
            model.fit(sample_data, resume=True)

    def test_resume_and_initial_centroids_raises(self, sample_data):
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        model.fit(sample_data)
        centroids = model.centroids_
        with pytest.raises(ValueError, match="Cannot use both"):
            model.fit(sample_data, initial_centroids=centroids, resume=True)


# ---- Error handling tests ----

class TestErrors:
    """Test error conditions."""

    def test_save_unfitted_raises(self, tmp_path):
        model = FCM(n_clusters=3, random_seed=42)
        with pytest.raises(NotFittedError):
            model.save(str(tmp_path / "bad.npz"))

    def test_load_unknown_class_raises(self, sample_data, tmp_path):
        """Tamper with metadata to test unknown class handling."""
        import json
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        model.fit(sample_data)
        path = str(tmp_path / "model.npz")
        model.save(path)

        # Load and tamper with metadata
        data = dict(np.load(path, allow_pickle=False))
        metadata = json.loads(str(data['__metadata__']))
        metadata['class_name'] = 'NonExistentModel'
        data['__metadata__'] = np.array(json.dumps(metadata))
        tampered_path = str(tmp_path / "tampered.npz")
        np.savez_compressed(tampered_path, **data)

        with pytest.raises(ValueError, match="Unknown model class"):
            FuzzyClusteringBase.load(tampered_path)


# ---- Two-phase model tests (IPCM, IPCM2, KIPCM, KIPCM2) ----

class TestTwoPhaseModels:
    """Test warm start and resume for two-phase models."""

    def test_ipcm_warm_start(self, sample_data):
        model = IPCM(n_clusters=3, max_iter=10, random_seed=42)
        centroids = sample_data[:3]
        model.fit(sample_data, initial_centroids=centroids)
        assert model.centroids_ is not None

    def test_ipcm_resume(self, sample_data):
        model = IPCM(n_clusters=3, max_iter=5, random_seed=42)
        model.fit(sample_data)
        model.max_iter = 20
        model.fit(sample_data, resume=True)
        assert model.centroids_ is not None

    def test_ipcm2_warm_start(self, sample_data):
        model = IPCM2(n_clusters=3, max_iter=10, random_seed=42)
        centroids = sample_data[:3]
        model.fit(sample_data, initial_centroids=centroids)
        assert model.centroids_ is not None

    def test_ipcm2_resume(self, sample_data):
        model = IPCM2(n_clusters=3, max_iter=5, random_seed=42)
        model.fit(sample_data)
        model.max_iter = 20
        model.fit(sample_data, resume=True)
        assert model.centroids_ is not None

    def test_ipcm_save_load(self, sample_data, tmp_path):
        model = IPCM(n_clusters=3, max_iter=10, random_seed=42)
        model.fit(sample_data)
        path = str(tmp_path / "ipcm.npz")
        model.save(path)
        loaded = FuzzyClusteringBase.load(path)
        assert type(loaded).__name__ == 'IPCM'
        np.testing.assert_array_almost_equal(
            np.asarray(loaded.centroids_), np.asarray(model.centroids_)
        )


# ---- Supervised resume-from-checkpoint tests ----

@pytest.fixture
def separable_2d():
    """Easy 2-class 2D dataset."""
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def iris_data():
    """3-class Iris dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.array(data.target, dtype=jnp.int32)
    return X, y


class TestSupervisedResume:
    """Test resume=True for supervised prototype models."""

    def test_glvq_resume_produces_valid_model(self, separable_2d):
        """Resume training should produce a working model with good accuracy."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        model.fit(X, y)

        # Resume for more iterations
        model.fit(X, y, resume=True)
        acc = float(jnp.mean(model.predict(X) == y))
        assert acc >= 0.75
        assert model.loss_ < float('inf')

    def test_glvq_resume_loss_starts_near_previous(self, separable_2d):
        """After resume, first loss should be close to where we left off."""
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.1, use_scan=False,
        )
        model.fit(X, y)
        loss_before = model.loss_history_[-1]

        model.fit(X, y, resume=True)
        loss_resume_start = model.loss_history_[0]
        # Resume should start near where we left off (within 10% absolute diff)
        assert abs(float(loss_resume_start) - float(loss_before)) < 0.1

    def test_gmlvq_resume_preserves_omega(self, iris_data):
        """GMLVQ resume should preserve omega dimensions."""
        X, y = iris_data
        model = GMLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
        )
        model.fit(X, y)
        omega_shape = model.omega_.shape

        model.fit(X, y, resume=True)
        assert model.omega_.shape == omega_shape
        assert model.omega_ is not None

    def test_grlvq_resume_preserves_relevances(self, iris_data):
        """GRLVQ resume should preserve relevance vector dimensions."""
        X, y = iris_data
        model = GRLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
        )
        model.fit(X, y)
        rel_shape = model.relevances_.shape

        model.fit(X, y, resume=True)
        assert model.relevances_.shape == rel_shape

    def test_resume_unfitted_raises(self, separable_2d):
        """Resume on unfitted model should raise NotFittedError."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        with pytest.raises(NotFittedError):
            model.fit(X, y, resume=True)

    def test_resume_with_initial_prototypes_raises(self, separable_2d):
        """resume=True + initial_prototypes should raise ValueError."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        model.fit(X, y)
        with pytest.raises(ValueError, match="Cannot use both"):
            model.fit(X, y, resume=True, initial_prototypes=model.prototypes_)

    def test_save_load_resume_roundtrip(self, separable_2d, tmp_path):
        """Save → load → resume should produce a working model."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.1)
        model.fit(X, y)
        path = str(tmp_path / "glvq.npz")
        model.save(path)

        loaded = GLVQ.load(path)
        loaded.fit(X, y, resume=True)
        acc = float(jnp.mean(loaded.predict(X) == y))
        assert acc >= 0.75

    def test_resume_scan_path(self, separable_2d):
        """Resume works with lax.scan training path."""
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.1, use_scan=True,
        )
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.loss_ < float('inf')

    def test_resume_minibatch_path(self, separable_2d):
        """Resume works with mini-batch training."""
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.1, batch_size=4,
        )
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.loss_ < float('inf')


class TestPartialFit:
    """Test partial_fit() for incremental/online learning."""

    def test_glvq_partial_fit_updates_prototypes(self, separable_2d):
        """partial_fit should update prototypes."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.1)
        model.fit(X, y)
        protos_before = model.prototypes_.copy()

        model.partial_fit(X, y)
        assert not jnp.array_equal(model.prototypes_, protos_before)

    def test_glvq_partial_fit_preserves_optimizer(self, separable_2d):
        """Optimizer state should persist across partial_fit calls."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.1)
        model.fit(X, y)

        # Multiple partial_fit calls
        losses = []
        for _ in range(10):
            model.partial_fit(X, y)
            losses.append(model.loss_)

        assert model._opt_state is not None
        assert len(losses) == 10

    def test_glvq_partial_fit_multiple_steps_improve(self, separable_2d):
        """Multiple partial_fit steps should maintain or improve quality."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=5, lr=0.1)
        model.fit(X, y)

        for _ in range(20):
            model.partial_fit(X, y)

        acc = float(jnp.mean(model.predict(X) == y))
        assert acc >= 0.75

    def test_gmlvq_partial_fit_preserves_omega(self, iris_data):
        """GMLVQ partial_fit should update omega too."""
        X, y = iris_data
        model = GMLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.01)
        model.fit(X, y)
        omega_before = model.omega_.copy()

        model.partial_fit(X, y)
        assert model.omega_ is not None
        assert model.omega_.shape == omega_before.shape

    def test_partial_fit_unfitted_raises(self, separable_2d):
        """partial_fit on unfitted model should raise NotFittedError."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        with pytest.raises(NotFittedError):
            model.partial_fit(X, y)

    def test_partial_fit_increments_n_iter(self, separable_2d):
        """n_iter_ should increment with each partial_fit call."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=10, lr=0.1)
        model.fit(X, y)
        n_iter_after_fit = model.n_iter_

        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_after_fit + 1

        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_after_fit + 2

    def test_save_load_partial_fit(self, separable_2d, tmp_path):
        """Save → load → partial_fit should work."""
        X, y = separable_2d
        model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.1)
        model.fit(X, y)
        path = str(tmp_path / "glvq.npz")
        model.save(path)

        loaded = GLVQ.load(path)
        loaded.partial_fit(X, y)
        acc = float(jnp.mean(loaded.predict(X) == y))
        assert acc >= 0.75

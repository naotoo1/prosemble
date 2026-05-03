"""
Tests for patience-based early stopping and restore_best in clustering models.

Tests cover:
- FCM with patience stops earlier than without
- restore_best returns state with lowest objective
- HCM patience works (Python-only loop model)
- KFCM patience works (kernel model)
- KohonenSOM patience works (use_scan forced to Python loop)
- Base class _check_patience logic
- Validation of patience parameter
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


@pytest.fixture
def clustered_data():
    """Three well-separated clusters for reliable convergence."""
    np.random.seed(42)
    X = jnp.concatenate([
        jnp.array(np.random.randn(30, 2) * 0.5 + [0, 0]),
        jnp.array(np.random.randn(30, 2) * 0.5 + [8, 8]),
        jnp.array(np.random.randn(30, 2) * 0.5 + [0, 8]),
    ])
    return X


@pytest.fixture
def two_cluster_data():
    """Two clear clusters."""
    np.random.seed(42)
    X = jnp.concatenate([
        jnp.array(np.random.randn(30, 2) * 0.5 + [0, 0]),
        jnp.array(np.random.randn(30, 2) * 0.5 + [6, 6]),
    ])
    return X


class TestCheckPatience:
    """Test the _check_patience helper in FuzzyClusteringBase."""

    def _make_model(self, **kwargs):
        """Create a minimal concrete subclass for testing base class methods."""
        from prosemble.models.base import FuzzyClusteringBase

        class _Concrete(FuzzyClusteringBase):
            def _iteration_step(self, state, X):
                return state, {}
            def predict(self, X):
                return X
            def _build_info(self, state, iteration):
                return {}

        return _Concrete(**kwargs)

    def test_not_enough_history(self):
        model = self._make_model(n_clusters=2, patience=5)
        assert model._check_patience([10, 9, 8], 5) is False

    def test_improving_does_not_trigger(self):
        model = self._make_model(n_clusters=2, patience=3)
        # Monotonically decreasing — should not trigger
        history = [10, 9, 8, 7, 6, 5]
        assert model._check_patience(history, 3) is False

    def test_plateau_triggers(self):
        model = self._make_model(n_clusters=2, patience=3)
        # Best was 5 at index 5, then 3 non-improving values
        history = [10, 8, 6, 5, 5.1, 5.2, 5.3]
        assert model._check_patience(history, 3) is True

    def test_validation_rejects_invalid(self):
        from prosemble.models.base import FuzzyClusteringBase
        with pytest.raises(ValueError, match="patience must be >= 1"):
            class _Concrete(FuzzyClusteringBase):
                def _iteration_step(self, state, X):
                    return state, {}
                def predict(self, X):
                    return X
                def _build_info(self, state, iteration):
                    return {}
            _Concrete(n_clusters=2, patience=0)


class TestFCMPatience:

    def test_patience_stops_earlier(self, clustered_data):
        from prosemble.models import FCM

        # Without patience — run all iterations
        fcm_full = FCM(n_clusters=3, max_iter=200, random_seed=42)
        fcm_full.fit(clustered_data)

        # With patience — should stop earlier (or at same if converged first)
        fcm_patience = FCM(n_clusters=3, max_iter=200, patience=5, random_seed=42)
        fcm_patience.fit(clustered_data)

        # Patience model should have stopped no later than full model
        assert fcm_patience.n_iter_ <= fcm_full.n_iter_

    def test_restore_best(self, clustered_data):
        from prosemble.models import FCM

        fcm = FCM(n_clusters=3, max_iter=50, restore_best=True, random_seed=42)
        fcm.fit(clustered_data)

        assert fcm.best_loss_ is not None
        assert fcm.best_loss_ <= float(fcm.objective_)

    def test_patience_with_restore_best(self, clustered_data):
        from prosemble.models import FCM

        fcm = FCM(
            n_clusters=3, max_iter=200, patience=10,
            restore_best=True, random_seed=42
        )
        fcm.fit(clustered_data)

        # Should have fitted successfully
        assert fcm.centroids_ is not None
        assert fcm.n_iter_ > 0


class TestHCMPatience:

    def test_patience_works(self, two_cluster_data):
        from prosemble.models import HCM

        hcm = HCM(n_clusters=2, max_iter=100, patience=5, random_seed=42)
        hcm.fit(two_cluster_data)

        assert hcm.centroids_ is not None
        assert hcm.n_iter_ > 0

    def test_restore_best(self, two_cluster_data):
        from prosemble.models import HCM

        hcm = HCM(n_clusters=2, max_iter=50, restore_best=True, random_seed=42)
        hcm.fit(two_cluster_data)

        assert hcm.best_loss_ is not None


class TestKFCMPatience:

    def test_patience_works(self, two_cluster_data):
        from prosemble.models import KFCM

        model = KFCM(n_clusters=2, max_iter=100, patience=5, random_seed=42)
        model.fit(two_cluster_data)

        assert model.centroids_ is not None
        assert model.n_iter_ > 0


class TestKohonenSOMPatience:

    def test_patience_forces_python_loop(self, two_cluster_data):
        from prosemble.models import KohonenSOM

        model = KohonenSOM(
            grid_height=3, grid_width=3, max_iter=100,
            patience=5, random_seed=42
        )
        model.fit(two_cluster_data)

        assert model.prototypes_ is not None
        assert model.n_iter_ > 0

    def test_restore_best(self, two_cluster_data):
        from prosemble.models import KohonenSOM

        model = KohonenSOM(
            grid_height=3, grid_width=3, max_iter=50,
            restore_best=True, random_seed=42, use_scan=False
        )
        model.fit(two_cluster_data)

        assert model.best_loss_ is not None


class TestHeskesSOMPatience:

    def test_patience_works(self, two_cluster_data):
        from prosemble.models import HeskesSOM

        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=100,
            patience=5, random_seed=42
        )
        model.fit(two_cluster_data)

        assert model.prototypes_ is not None
        assert model.n_iter_ > 0


class TestPCMPatience:

    def test_patience_works(self, two_cluster_data):
        from prosemble.models import PCM

        model = PCM(n_clusters=2, max_iter=100, patience=5, random_seed=42)
        model.fit(two_cluster_data)

        assert model.centroids_ is not None
        assert model.n_iter_ > 0

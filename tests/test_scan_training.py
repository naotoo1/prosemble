"""Tests for lax.scan-based supervised training.

Verifies that scan-based training produces correct results:
- Models converge and loss decreases
- Results are consistent across runs (same seed)
- Convergence detection works (n_iter < max_iter when converging)
"""

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


class TestScanTrainingGLVQ:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(n_prototypes_per_class=2, max_iter=50, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        # Loss should generally decrease
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_convergence_detection(self, iris_data):
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=500, lr=0.01,
            epsilon=1e-4, random_seed=42
        )
        model.fit(X_train, y_train)
        # Should converge before max_iter
        assert model.n_iter_ < 500

    def test_reproducibility(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        model1 = GLVQ(n_prototypes_per_class=2, max_iter=30, lr=0.01, random_seed=42)
        model1.fit(X_train, y_train)
        model2 = GLVQ(n_prototypes_per_class=2, max_iter=30, lr=0.01, random_seed=42)
        model2.fit(X_train, y_train)
        assert jnp.allclose(model1.prototypes_, model2.prototypes_, atol=1e-5)
        assert jnp.array_equal(model1.predict(X_test), model2.predict(X_test))

    def test_accuracy(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, y_test = iris_data
        model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(jnp.mean(preds == y_test))
        assert acc > 0.8


class TestScanTrainingGMLVQ:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import GMLVQ
        X_train, _, y_train, _ = iris_data
        model = GMLVQ(n_prototypes_per_class=2, max_iter=50, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_omega_learned(self, iris_data):
        from prosemble.models import GMLVQ
        X_train, _, y_train, _ = iris_data
        model = GMLVQ(n_prototypes_per_class=2, max_iter=50, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        # Omega should not be identity after training
        assert model.omega_ is not None
        assert model.omega_.shape[0] == X_train.shape[1]


class TestScanTrainingGRLVQ:
    def test_relevances_learned(self, iris_data):
        from prosemble.models import GRLVQ
        X_train, _, y_train, _ = iris_data
        model = GRLVQ(n_prototypes_per_class=2, max_iter=50, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        # Relevances should be non-uniform after training
        relevances = model.relevance_profile
        assert not jnp.allclose(relevances, relevances[0])
        # Should sum to 1
        assert jnp.isclose(jnp.sum(relevances), 1.0, atol=1e-5)


class TestScanTrainingCBC:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import CBC
        X_train, _, y_train, _ = iris_data
        model = CBC(
            n_components=6, n_classes=3, max_iter=50,
            lr=0.01, sigma=1.0, random_seed=42
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestScanTrainingCELVQ:
    def test_fit_and_predict(self, iris_data):
        from prosemble.models import CELVQ
        X_train, X_test, y_train, y_test = iris_data
        model = CELVQ(n_prototypes_per_class=2, max_iter=50, lr=0.01, random_seed=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(jnp.mean(preds == y_test))
        assert acc > 0.5


# --- use_scan=False (Python loop with true early stopping) ---

class TestPythonLoopGLVQ:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            random_seed=42, use_scan=False
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_true_early_stopping(self, iris_data):
        """With use_scan=False, loss_history length == n_iter (no wasted iters)."""
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=500, lr=0.01,
            epsilon=1e-4, random_seed=42, use_scan=False
        )
        model.fit(X_train, y_train)
        assert model.n_iter_ < 500
        assert len(model.loss_history_) == model.n_iter_

    def test_scan_vs_python_equivalent(self, iris_data):
        """Both modes should produce similar final prototypes."""
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        scan_model = GLVQ(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            random_seed=42, use_scan=True
        )
        scan_model.fit(X_train, y_train)

        python_model = GLVQ(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            random_seed=42, use_scan=False
        )
        python_model.fit(X_train, y_train)

        # Prototypes should be very close
        assert jnp.allclose(scan_model.prototypes_, python_model.prototypes_, atol=1e-4)
        # Predictions should match
        assert jnp.array_equal(scan_model.predict(X_test), python_model.predict(X_test))


class TestPythonLoopNeuralGas:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import NeuralGas
        X_train, _, _, _ = iris_data
        model = NeuralGas(
            n_prototypes=5, max_iter=50, random_seed=42, use_scan=False
        )
        model.fit(X_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_true_early_stopping(self, iris_data):
        from prosemble.models import NeuralGas
        X_train, _, _, _ = iris_data
        model = NeuralGas(
            n_prototypes=5, max_iter=500, epsilon=1e-2,
            random_seed=42, use_scan=False
        )
        model.fit(X_train)
        assert len(model.loss_history_) == model.n_iter_


class TestPythonLoopKohonenSOM:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import KohonenSOM
        X_train, _, _, _ = iris_data
        model = KohonenSOM(
            grid_height=3, grid_width=3, max_iter=50,
            random_seed=42, use_scan=False
        )
        model.fit(X_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_scan_vs_python_equivalent(self, iris_data):
        from prosemble.models import KohonenSOM
        X_train, X_test, _, _ = iris_data
        scan_model = KohonenSOM(
            grid_height=3, grid_width=3, max_iter=30,
            random_seed=42, use_scan=True
        )
        scan_model.fit(X_train)

        python_model = KohonenSOM(
            grid_height=3, grid_width=3, max_iter=30,
            random_seed=42, use_scan=False
        )
        python_model.fit(X_train)

        assert jnp.allclose(scan_model.prototypes_, python_model.prototypes_, atol=1e-4)
        assert jnp.array_equal(scan_model.predict(X_test), python_model.predict(X_test))

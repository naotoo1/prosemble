"""Tests for mini-batch training.

Verifies that mini-batch training:
- Converges (loss decreases)
- Produces reasonable accuracy
- Works with both scan and Python loop modes
- Matches full-batch results approximately
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


class TestMiniBatchScanGLVQ:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            batch_size=32, random_seed=42,
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_accuracy(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, y_test = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=100, lr=0.01,
            batch_size=32, random_seed=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(jnp.mean(preds == y_test))
        assert acc > 0.7

    def test_reproducibility(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        kwargs = dict(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            batch_size=32, random_seed=42,
        )
        model1 = GLVQ(**kwargs)
        model1.fit(X_train, y_train)
        model2 = GLVQ(**kwargs)
        model2.fit(X_train, y_train)
        assert jnp.allclose(model1.prototypes_, model2.prototypes_, atol=1e-5)


class TestMiniBatchPythonGLVQ:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            batch_size=32, random_seed=42, use_scan=False,
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_true_early_stopping(self, iris_data):
        """With use_scan=False, loss_history length == n_iter."""
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=500, lr=0.01,
            batch_size=32, epsilon=1e-4, random_seed=42, use_scan=False,
        )
        model.fit(X_train, y_train)
        assert len(model.loss_history_) == model.n_iter_

    def test_accuracy(self, iris_data):
        from prosemble.models import GLVQ
        X_train, X_test, y_train, y_test = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=100, lr=0.01,
            batch_size=32, random_seed=42, use_scan=False,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(jnp.mean(preds == y_test))
        assert acc > 0.7


class TestMiniBatchScanGMLVQ:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import GMLVQ
        X_train, _, y_train, _ = iris_data
        model = GMLVQ(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            batch_size=32, random_seed=42,
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestMiniBatchScanCBC:
    def test_loss_decreases(self, iris_data):
        from prosemble.models import CBC
        X_train, _, y_train, _ = iris_data
        model = CBC(
            n_components=6, n_classes=3, max_iter=50,
            lr=0.01, sigma=1.0, batch_size=32, random_seed=42,
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestMiniBatchBatchSizeVariants:
    def test_batch_size_1(self, iris_data):
        """Batch size 1 = online/stochastic gradient descent."""
        from prosemble.models import GLVQ
        X_train, _, y_train, _ = iris_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=20, lr=0.001,
            batch_size=1, random_seed=42, use_scan=False,
        )
        model.fit(X_train, y_train)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_batch_size_equals_dataset(self, iris_data):
        """Batch size == n_samples should behave like full-batch."""
        from prosemble.models import GLVQ
        X_train, X_test, y_train, _ = iris_data
        n_samples = X_train.shape[0]

        full_model = GLVQ(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            random_seed=42, use_scan=False,
        )
        full_model.fit(X_train, y_train)

        batch_model = GLVQ(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            batch_size=n_samples, random_seed=42, use_scan=False,
        )
        batch_model.fit(X_train, y_train)

        # Should produce similar (not identical due to shuffle) results
        full_preds = full_model.predict(X_test)
        batch_preds = batch_model.predict(X_test)
        agreement = float(jnp.mean(full_preds == batch_preds))
        assert agreement > 0.8

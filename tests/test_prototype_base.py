"""Tests for prototype base classes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.prototype_base import (
    SupervisedPrototypeModel,
    UnsupervisedPrototypeModel,
    NotFittedError,
)


# --- Concrete subclass for testing ---

class SimpleGLVQ(SupervisedPrototypeModel):
    """Minimal GLVQ for testing the base class."""

    def _compute_loss(self, params, X, y, proto_labels):
        from prosemble.core.losses import glvq_loss
        distances = self.distance_fn(X, params['prototypes'])
        return glvq_loss(distances, y, proto_labels, margin=self.margin)


class SimpleUnsupervised(UnsupervisedPrototypeModel):
    """Minimal unsupervised model for testing."""

    def fit(self, X):
        X = jnp.asarray(X, dtype=jnp.float32)
        # Just initialize prototypes randomly from data
        key = self.key
        indices = jax.random.choice(key, X.shape[0], (self.n_prototypes,), replace=False)
        self.prototypes_ = X[indices]
        self.n_iter_ = 1
        self.loss_ = 0.0
        self.loss_history_ = jnp.array([0.0])
        return self


# --- Tests for SupervisedPrototypeModel ---

class TestSupervisedInit:
    def test_default_params(self):
        model = SimpleGLVQ()
        assert model.n_prototypes_per_class == 1
        assert model.max_iter == 100
        assert model.lr == 0.01

    def test_invalid_n_prototypes(self):
        with pytest.raises(ValueError):
            SimpleGLVQ(n_prototypes_per_class=0)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError):
            SimpleGLVQ(max_iter=0)

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            SimpleGLVQ(lr=-0.1)

    def test_not_fitted(self):
        model = SimpleGLVQ()
        with pytest.raises(NotFittedError):
            model.predict(jnp.ones((3, 2)))


class TestSupervisedFit:
    @pytest.fixture
    def simple_data(self):
        """Linearly separable 2-class dataset."""
        X = jnp.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
            [1.0, 1.0], [1.1, 1.1], [1.2, 1.0],
        ])
        y = jnp.array([0, 0, 0, 1, 1, 1])
        return X, y

    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(max_iter=10)
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_attributes(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(max_iter=10)
        model.fit(X, y)

        assert model.prototypes_ is not None
        assert model.prototype_labels_ is not None
        assert model.classes_ is not None
        assert model.n_classes_ == 2
        assert model.n_iter_ is not None
        assert model.loss_history_ is not None

    def test_prototype_shapes(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(n_prototypes_per_class=2, max_iter=10)
        model.fit(X, y)

        assert model.prototypes_.shape == (4, 2)  # 2 classes * 2 per class
        assert model.prototype_labels_.shape == (4,)

    def test_loss_decreases(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(max_iter=50, lr=0.1)
        model.fit(X, y)

        history = model.loss_history_
        # Loss should generally decrease (allow some noise)
        assert history[-1] < history[0]

    def test_predict_shape(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(max_iter=20, lr=0.1)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (6,)

    def test_predict_proba_shape(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(max_iter=20, lr=0.1)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (6, 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_accuracy_on_simple_data(self, simple_data):
        X, y = simple_data
        model = SimpleGLVQ(max_iter=100, lr=0.1)
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = jnp.mean(preds == y)
        assert accuracy >= 0.8  # Should get most right on easy data

    def test_invalid_X_shape(self):
        model = SimpleGLVQ()
        with pytest.raises(ValueError, match="2D"):
            model.fit(jnp.ones((10,)), jnp.zeros(10))

    def test_initial_prototypes(self, simple_data):
        X, y = simple_data
        init_protos = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        model = SimpleGLVQ(max_iter=5)
        model.fit(X, y, initial_prototypes=init_protos)
        assert model.prototypes_ is not None


class TestSupervisedSerialization:
    def test_save_load(self):
        X = jnp.array([
            [0.0, 0.0], [0.1, 0.1],
            [1.0, 1.0], [1.1, 1.1],
        ])
        y = jnp.array([0, 0, 1, 1])

        model = SimpleGLVQ(max_iter=10)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name

        try:
            model.save(path)
            assert os.path.exists(path)

            # Verify file contents
            data = np.load(path, allow_pickle=False)
            assert '__metadata__' in data
            assert 'prototypes_' in data
            assert 'prototype_labels_' in data
        finally:
            os.unlink(path)

    def test_not_fitted_save(self):
        model = SimpleGLVQ()
        with pytest.raises(NotFittedError):
            model.save('/tmp/test.npz')


# --- Tests for UnsupervisedPrototypeModel ---

class TestUnsupervisedInit:
    def test_default_params(self):
        model = SimpleUnsupervised(n_prototypes=5)
        assert model.n_prototypes == 5
        assert model.max_iter == 100

    def test_invalid_n_prototypes(self):
        with pytest.raises(ValueError):
            SimpleUnsupervised(n_prototypes=0)

    def test_not_fitted(self):
        model = SimpleUnsupervised(n_prototypes=3)
        with pytest.raises(NotFittedError):
            model.predict(jnp.ones((3, 2)))


class TestUnsupervisedFit:
    def test_fit_sets_prototypes(self):
        X = jnp.ones((10, 3))
        model = SimpleUnsupervised(n_prototypes=3)
        model.fit(X)
        assert model.prototypes_.shape == (3, 3)

    def test_predict_shape(self):
        X = jnp.ones((10, 3))
        model = SimpleUnsupervised(n_prototypes=3)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (10,)

    def test_transform_shape(self):
        X = jnp.ones((10, 3))
        model = SimpleUnsupervised(n_prototypes=3)
        model.fit(X)
        dists = model.transform(X)
        assert dists.shape == (10, 3)

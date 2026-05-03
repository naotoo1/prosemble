"""Tests for LVQMLN and PLVQ models."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.datasets import load_iris_jax
from prosemble.models import LVQMLN, PLVQ


@pytest.fixture(scope="module")
def iris():
    dataset = load_iris_jax()
    return dataset.input_data, dataset.labels


# ---------------------------------------------------------------------------
# LVQMLN
# ---------------------------------------------------------------------------

class TestLVQMLNBasic:
    def test_fit(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.backbone_params_ is not None
        # Prototypes should be in latent space (dim=2)
        assert model.prototypes_.shape == (3, 2)

    def test_predict(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (150,)
        assert jnp.all(preds >= 0) and jnp.all(preds < 3)

    def test_predict_proba(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (150, 3)
        # Probabilities sum to 1
        assert jnp.allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_transform(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        latent = model.transform(X)
        assert latent.shape == (150, 3)

    def test_loss_decreases(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=2, max_iter=100,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) < float(history[0])


class TestLVQMLNArchitecture:
    def test_deeper_network(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[16, 8], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_.shape == (3, 2)
        # Backbone has 2 layers
        assert len(model.backbone_params_) == 3  # 2 hidden + 1 output

    def test_relu_activation(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            activation='relu',
            n_prototypes_per_class=1, max_iter=30,
            lr=0.001, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None

    def test_tanh_activation(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            activation='tanh',
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None

    def test_multiple_prototypes_per_class(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=3, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 2)


class TestLVQMLNReproducibility:
    def test_same_seed(self, iris):
        X, y = iris
        m1 = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m1.fit(X, y)
        m2 = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m2.fit(X, y)
        assert jnp.allclose(m1.prototypes_, m2.prototypes_, atol=1e-5)


class TestLVQMLNDualMode:
    def test_scan_mode(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42, use_scan=True,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None

    def test_python_mode(self, iris):
        X, y = iris
        model = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None


# ---------------------------------------------------------------------------
# PLVQ
# ---------------------------------------------------------------------------

class TestPLVQBasic:
    def test_fit(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.backbone_params_ is not None
        assert model.prototypes_.shape == (3, 2)

    def test_predict(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (150,)
        assert jnp.all(preds >= 0) and jnp.all(preds < 3)

    def test_predict_proba(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=2, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (150, 3)
        # Probabilities should sum to ~1
        sums = jnp.sum(proba, axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-4)

    def test_transform(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        latent = model.transform(X)
        assert latent.shape == (150, 3)

    def test_loss_decreases(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=2, max_iter=100,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) < float(history[0])


class TestPLVQLossTypes:
    def test_rslvq_loss(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=2, loss_type='rslvq',
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None

    def test_nllr_loss(self, iris):
        X, y = iris
        model = PLVQ(
            hidden_sizes=[10], latent_dim=2, loss_type='nllr',
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None


class TestPLVQReproducibility:
    def test_same_seed(self, iris):
        X, y = iris
        m1 = PLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m1.fit(X, y)
        m2 = PLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m2.fit(X, y)
        assert jnp.allclose(m1.prototypes_, m2.prototypes_, atol=1e-5)

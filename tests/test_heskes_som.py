"""Tests for Heskes SOM."""

import jax.numpy as jnp
import pytest

from prosemble.datasets import load_iris_jax
from prosemble.models import HeskesSOM


@pytest.fixture(scope="module")
def iris_X():
    dataset = load_iris_jax()
    return dataset.input_data


class TestHeskesSOMBasic:
    def test_fit(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30, random_seed=42
        )
        model.fit(iris_X)
        assert model.prototypes_.shape == (9, 4)
        assert model.n_iter_ <= 30

    def test_predict(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30, random_seed=42
        )
        model.fit(iris_X)
        labels = model.predict(iris_X)
        assert labels.shape == (150,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 9)

    def test_transform(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30, random_seed=42
        )
        model.fit(iris_X)
        dists = model.transform(iris_X)
        assert dists.shape == (150, 9)
        assert jnp.all(dists >= 0)

    def test_bmu_map(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30, random_seed=42
        )
        model.fit(iris_X)
        coords = model.bmu_map(iris_X)
        assert coords.shape == (150, 2)
        assert jnp.all(coords[:, 0] >= 0)
        assert jnp.all(coords[:, 0] < 3)
        assert jnp.all(coords[:, 1] >= 0)
        assert jnp.all(coords[:, 1] < 3)


class TestHeskesSOMEnergy:
    def test_energy_decreases(self, iris_X):
        """Heskes SOM guarantees monotonic energy decrease."""
        model = HeskesSOM(
            grid_height=4, grid_width=4, max_iter=50, random_seed=42
        )
        model.fit(iris_X)
        history = model.loss_history_[:model.n_iter_]
        diffs = jnp.diff(history)
        # Energy should never increase (within numerical tolerance)
        assert jnp.all(diffs <= 1e-4)

    def test_loss_decreases(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=50, random_seed=42
        )
        model.fit(iris_X)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestHeskesSOMDualMode:
    def test_scan_mode(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30,
            random_seed=42, use_scan=True
        )
        model.fit(iris_X)
        assert model.prototypes_ is not None

    def test_python_mode(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30,
            random_seed=42, use_scan=False
        )
        model.fit(iris_X)
        assert model.prototypes_ is not None

    def test_scan_vs_python_equivalent(self, iris_X):
        scan_model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30,
            random_seed=42, use_scan=True
        )
        scan_model.fit(iris_X)

        python_model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=30,
            random_seed=42, use_scan=False
        )
        python_model.fit(iris_X)

        assert jnp.allclose(
            scan_model.prototypes_, python_model.prototypes_, atol=1e-4
        )
        assert jnp.array_equal(
            scan_model.predict(iris_X), python_model.predict(iris_X)
        )

    def test_python_true_early_stopping(self, iris_X):
        model = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=500,
            epsilon=1e-2, random_seed=42, use_scan=False
        )
        model.fit(iris_X)
        assert len(model.loss_history_) == model.n_iter_


class TestHeskesSOMReproducibility:
    def test_same_seed(self, iris_X):
        m1 = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=20, random_seed=42
        )
        m1.fit(iris_X)
        m2 = HeskesSOM(
            grid_height=3, grid_width=3, max_iter=20, random_seed=42
        )
        m2.fit(iris_X)
        assert jnp.allclose(m1.prototypes_, m2.prototypes_, atol=1e-6)

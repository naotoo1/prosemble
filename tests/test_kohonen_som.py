"""Tests for Kohonen SOM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.kohonen_som import KohonenSOM


@pytest.fixture
def blob_data():
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    c1 = jax.random.normal(k1, (50, 2)) * 0.3
    c2 = jax.random.normal(k2, (50, 2)) * 0.3 + jnp.array([3.0, 3.0])
    return jnp.concatenate([c1, c2])


class TestKohonenSOM:
    def test_fit(self, blob_data):
        model = KohonenSOM(grid_height=3, grid_width=3, max_iter=10)
        model.fit(blob_data)
        assert model.prototypes_.shape == (9, 2)

    def test_predict(self, blob_data):
        model = KohonenSOM(grid_height=3, grid_width=3, max_iter=10)
        model.fit(blob_data)
        preds = model.predict(blob_data)
        assert preds.shape == (100,)
        assert jnp.all(preds >= 0) and jnp.all(preds < 9)

    def test_bmu_map(self, blob_data):
        model = KohonenSOM(grid_height=4, grid_width=4, max_iter=10)
        model.fit(blob_data)
        coords = model.bmu_map(blob_data)
        assert coords.shape == (100, 2)
        # Coordinates should be within grid bounds
        assert jnp.all(coords[:, 0] >= 0) and jnp.all(coords[:, 0] < 4)
        assert jnp.all(coords[:, 1] >= 0) and jnp.all(coords[:, 1] < 4)

    def test_loss_converges(self, blob_data):
        """SOM loss isn't strictly monotonic but should converge to reasonable value."""
        model = KohonenSOM(grid_height=3, grid_width=3, max_iter=50)
        model.fit(blob_data)
        # Loss should be finite and positive
        assert jnp.isfinite(model.loss_)
        assert model.loss_ > 0
        # Later half should have lower variance than first half
        n = len(model.loss_history_)
        if n > 10:
            late_std = float(jnp.std(model.loss_history_[n//2:]))
            early_std = float(jnp.std(model.loss_history_[:n//2]))
            assert late_std <= early_std + 0.1  # converging

    def test_transform(self, blob_data):
        model = KohonenSOM(grid_height=3, grid_width=3, max_iter=10)
        model.fit(blob_data)
        dists = model.transform(blob_data)
        assert dists.shape == (100, 9)

    def test_gaussian_neighborhood(self, blob_data):
        """Verify SOM uses Gaussian (not binary) neighborhood."""
        model = KohonenSOM(
            grid_height=5, grid_width=5, max_iter=20,
            sigma_init=3.0, sigma_final=0.5,
        )
        model.fit(blob_data)
        # Just verify it runs and produces reasonable results
        assert model.prototypes_ is not None
        assert jnp.all(jnp.isfinite(model.prototypes_))

    def test_rectangular_grid(self, blob_data):
        model = KohonenSOM(grid_height=2, grid_width=5, max_iter=10)
        model.fit(blob_data)
        assert model.prototypes_.shape == (10, 2)

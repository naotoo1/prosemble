"""Tests for Neural Gas and Growing Neural Gas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.neural_gas import NeuralGas
from prosemble.models.growing_neural_gas import GrowingNeuralGas


@pytest.fixture
def blob_data():
    """Two clusters of 2D data."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    c1 = jax.random.normal(k1, (50, 2)) * 0.3 + jnp.array([0.0, 0.0])
    c2 = jax.random.normal(k2, (50, 2)) * 0.3 + jnp.array([3.0, 3.0])
    return jnp.concatenate([c1, c2])


class TestNeuralGas:
    def test_fit(self, blob_data):
        model = NeuralGas(n_prototypes=5, max_iter=20, lr_init=0.5, lr_final=0.01)
        model.fit(blob_data)
        assert model.prototypes_.shape == (5, 2)

    def test_predict(self, blob_data):
        model = NeuralGas(n_prototypes=5, max_iter=20)
        model.fit(blob_data)
        preds = model.predict(blob_data)
        assert preds.shape == (100,)
        assert jnp.all(preds >= 0) and jnp.all(preds < 5)

    def test_transform(self, blob_data):
        model = NeuralGas(n_prototypes=5, max_iter=20)
        model.fit(blob_data)
        dists = model.transform(blob_data)
        assert dists.shape == (100, 5)

    def test_loss_decreases(self, blob_data):
        model = NeuralGas(n_prototypes=5, max_iter=50)
        model.fit(blob_data)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_lambda_decay(self, blob_data):
        model = NeuralGas(
            n_prototypes=5, max_iter=20,
            lambda_init=5.0, lambda_final=0.01,
        )
        model.fit(blob_data)
        assert model.prototypes_ is not None


class TestGrowingNeuralGas:
    def test_fit(self, blob_data):
        model = GrowingNeuralGas(
            max_nodes=20, max_iter=5,
            insert_interval=50, lr_winner=0.1,
        )
        model.fit(blob_data)
        assert model.prototypes_ is not None
        assert model.n_active_ >= 2

    def test_grows(self, blob_data):
        model = GrowingNeuralGas(
            max_nodes=20, max_iter=10,
            insert_interval=20, lr_winner=0.1,
        )
        model.fit(blob_data)
        # Should have grown beyond initial 2
        assert model.n_active_ > 2

    def test_predict(self, blob_data):
        model = GrowingNeuralGas(max_nodes=10, max_iter=5, insert_interval=50)
        model.fit(blob_data)
        preds = model.predict(blob_data)
        assert preds.shape == (100,)

    def test_max_nodes_respected(self, blob_data):
        model = GrowingNeuralGas(
            max_nodes=5, max_iter=20,
            insert_interval=10,
        )
        model.fit(blob_data)
        assert model.n_active_ <= 5

    def test_edges(self, blob_data):
        model = GrowingNeuralGas(max_nodes=10, max_iter=5, insert_interval=50)
        model.fit(blob_data)
        assert model.edges_ is not None
        assert model.edges_.shape[0] == model.n_active_

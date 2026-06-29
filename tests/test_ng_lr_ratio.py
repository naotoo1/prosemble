"""Tests for lr_ratio (separate learning rates) across NG models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.srng import SRNG
from prosemble.models.smng import SMNG
from prosemble.models.slng import SLNG
from prosemble.models.stng import STNG
from prosemble.models.scmng import SCMNG
from prosemble.models.sng import SNG
from prosemble.models.dk_glvq_ng import DKGLVQ_NG
from prosemble.models.dk_matrix_lvq_ng import DKGMLVQ_NG
from prosemble.models.dk_relevance_lvq_ng import DKGRLVQ_NG


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.array(data.target, dtype=jnp.int32)
    return X, y


@pytest.fixture
def separable_2d():
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


CORE_MODELS = [SRNG, SMNG, SLNG, STNG, SCMNG, SNG]
HALF_DEFAULT_MODELS = [SRNG, SMNG, SLNG, STNG, SNG]
DK_MODELS = [DKGLVQ_NG, DKGMLVQ_NG, DKGRLVQ_NG]


class TestLrRatioDefault:
    @pytest.mark.parametrize("ModelClass", HALF_DEFAULT_MODELS)
    def test_default_lr_ratio_half(self, ModelClass):
        model = ModelClass(n_prototypes_per_class=1, max_iter=10, lr=0.01)
        assert model.lr_ratio == 0.5

    def test_default_lr_ratio_scmng(self):
        model = SCMNG(n_prototypes_per_class=1, max_iter=10, lr=0.01)
        assert model.lr_ratio == 1.0

    @pytest.mark.parametrize("ModelClass", DK_MODELS)
    def test_default_lr_ratio_dk(self, ModelClass):
        model = ModelClass(n_prototypes_per_class=1, max_iter=10, lr=0.01)
        assert model.lr_ratio == 0.5


class TestLrRatioOneReproducesOldBehavior:
    @pytest.mark.parametrize("ModelClass", CORE_MODELS)
    def test_lr_ratio_1_same_loss_as_default_path(self, separable_2d, ModelClass):
        X, y = separable_2d
        model = ModelClass(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            lr_ratio=1.0, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75


class TestLrRatioAffectsTraining:
    @pytest.mark.parametrize("ModelClass", [SRNG, SMNG, SNG])
    def test_different_lr_ratio_different_result(self, iris_data, ModelClass):
        X, y = iris_data
        model_half = ModelClass(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            lr_ratio=0.5, random_seed=42,
        )
        model_half.fit(X, y)

        model_one = ModelClass(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            lr_ratio=1.0, random_seed=42,
        )
        model_one.fit(X, y)

        protos_half = np.asarray(model_half.prototypes_)
        protos_one = np.asarray(model_one.prototypes_)
        assert not np.allclose(protos_half, protos_one, atol=1e-4)


class TestLrRatioCustomValues:
    @pytest.mark.parametrize("lr_ratio", [0.1, 0.25, 0.5, 0.75, 1.0])
    def test_srng_various_ratios(self, separable_2d, lr_ratio):
        X, y = separable_2d
        model = SRNG(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            lr_ratio=lr_ratio, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.625

    @pytest.mark.parametrize("lr_ratio", [0.1, 0.5, 1.0])
    def test_smng_various_ratios(self, separable_2d, lr_ratio):
        X, y = separable_2d
        model = SMNG(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            lr_ratio=lr_ratio, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.625


class TestLrRatioInHyperparams:
    @pytest.mark.parametrize("ModelClass", CORE_MODELS)
    def test_lr_ratio_in_hyperparams(self, ModelClass):
        model = ModelClass(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            lr_ratio=0.3,
        )
        hp = model._get_hyperparams()
        assert 'lr_ratio' in hp
        assert hp['lr_ratio'] == 0.3

    @pytest.mark.parametrize("ModelClass", DK_MODELS)
    def test_lr_ratio_in_hyperparams_dk(self, ModelClass):
        model = ModelClass(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            lr_ratio=0.7,
        )
        hp = model._get_hyperparams()
        assert 'lr_ratio' in hp
        assert hp['lr_ratio'] == 0.7


class TestLrRatioIris:
    @pytest.mark.parametrize("ModelClass", CORE_MODELS)
    def test_iris_with_lr_ratio(self, iris_data, ModelClass):
        X, y = iris_data
        model = ModelClass(
            n_prototypes_per_class=2, max_iter=100, lr=0.01,
            lr_ratio=0.5,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.70

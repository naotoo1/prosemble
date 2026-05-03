"""Tests for Siamese LVQ models."""

import jax.numpy as jnp
import pytest

from prosemble.datasets import load_iris_jax
from prosemble.models import SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ


@pytest.fixture(scope="module")
def iris():
    dataset = load_iris_jax()
    return dataset.input_data, dataset.labels


# ---------------------------------------------------------------------------
# SiameseGLVQ
# ---------------------------------------------------------------------------

class TestSiameseGLVQ:
    def test_fit(self, iris):
        X, y = iris
        model = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.backbone_params_ is not None
        # Prototypes in input space (4D for Iris)
        assert model.prototypes_.shape == (3, 4)

    def test_predict(self, iris):
        X, y = iris
        model = SiameseGLVQ(
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
        model = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (150, 3)
        assert jnp.allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_transform(self, iris):
        X, y = iris
        model = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        latent = model.transform(X)
        assert latent.shape == (150, 3)

    def test_loss_decreases(self, iris):
        X, y = iris
        model = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=2, max_iter=100,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])

    def test_reproducibility(self, iris):
        X, y = iris
        m1 = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m1.fit(X, y)
        m2 = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m2.fit(X, y)
        assert jnp.allclose(m1.prototypes_, m2.prototypes_, atol=1e-5)

    def test_scan_mode(self, iris):
        X, y = iris
        model = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42, use_scan=True,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None


# ---------------------------------------------------------------------------
# SiameseGMLVQ
# ---------------------------------------------------------------------------

class TestSiameseGMLVQ:
    def test_fit(self, iris):
        X, y = iris
        model = SiameseGMLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.omega_ is not None
        assert model.backbone_params_ is not None
        # Prototypes in input space
        assert model.prototypes_.shape == (3, 4)

    def test_predict(self, iris):
        X, y = iris
        model = SiameseGMLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (150,)

    def test_lambda_matrix(self, iris):
        X, y = iris
        model = SiameseGMLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (3, 3)

    def test_loss_decreases(self, iris):
        X, y = iris
        model = SiameseGMLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=100,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])

    def test_reproducibility(self, iris):
        X, y = iris
        m1 = SiameseGMLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m1.fit(X, y)
        m2 = SiameseGMLVQ(
            hidden_sizes=[10], latent_dim=3,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m2.fit(X, y)
        assert jnp.allclose(m1.prototypes_, m2.prototypes_, atol=1e-5)


# ---------------------------------------------------------------------------
# SiameseGTLVQ
# ---------------------------------------------------------------------------

class TestSiameseGTLVQ:
    def test_fit(self, iris):
        X, y = iris
        model = SiameseGTLVQ(
            hidden_sizes=[10], latent_dim=4, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.omegas_ is not None
        assert model.backbone_params_ is not None
        # Prototypes in input space
        assert model.prototypes_.shape == (3, 4)
        # Omegas in latent space
        assert model.omegas_.shape == (3, 4, 2)

    def test_predict(self, iris):
        X, y = iris
        model = SiameseGTLVQ(
            hidden_sizes=[10], latent_dim=4, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (150,)

    def test_loss_decreases(self, iris):
        X, y = iris
        model = SiameseGTLVQ(
            hidden_sizes=[10], latent_dim=4, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=100,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])

    def test_reproducibility(self, iris):
        X, y = iris
        m1 = SiameseGTLVQ(
            hidden_sizes=[10], latent_dim=4, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m1.fit(X, y)
        m2 = SiameseGTLVQ(
            hidden_sizes=[10], latent_dim=4, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        m2.fit(X, y)
        assert jnp.allclose(m1.prototypes_, m2.prototypes_, atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

class TestSiameseVsLVQMLN:
    """Verify Siamese models keep prototypes in input space,
    while LVQMLN keeps them in latent space."""

    def test_prototype_space_difference(self, iris):
        X, y = iris
        from prosemble.models import LVQMLN

        # LVQMLN: prototypes in latent space (dim=2)
        lvqmln = LVQMLN(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        lvqmln.fit(X, y)
        assert lvqmln.prototypes_.shape[1] == 2  # latent dim

        # SiameseGLVQ: prototypes in input space (dim=4)
        siamese = SiameseGLVQ(
            hidden_sizes=[10], latent_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        siamese.fit(X, y)
        assert siamese.prototypes_.shape[1] == 4  # input dim

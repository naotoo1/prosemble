"""Tests for Riemannian Neural Gas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.manifolds import SO, SPD, Grassmannian
from prosemble.models import RiemannianNeuralGas


# ---------------------------------------------------------------------------
# Manifold primitive tests
# ---------------------------------------------------------------------------

class TestSOManifold:
    def test_random_point_is_rotation(self):
        manifold = SO(3)
        key = jax.random.PRNGKey(0)
        R = manifold.random_point(key)
        assert R.shape == (3, 3)
        assert manifold.belongs(R)

    def test_exp_log_roundtrip(self):
        manifold = SO(3)
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        R = manifold.random_point(k1)
        S = manifold.random_point(k2)
        V = manifold.log_map(R, S)
        S_recovered = manifold.exp_map(R, V)
        np.testing.assert_allclose(S_recovered, S, atol=1e-3)

    def test_distance_self_is_zero(self):
        manifold = SO(3)
        R = manifold.random_point(jax.random.PRNGKey(0))
        d = manifold.distance(R, R)
        assert float(d) < 1e-5

    def test_distance_symmetric(self):
        manifold = SO(3)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        R = manifold.random_point(k1)
        S = manifold.random_point(k2)
        np.testing.assert_allclose(
            manifold.distance(R, S), manifold.distance(S, R), atol=1e-4
        )

    def test_project_to_so(self):
        manifold = SO(3)
        A = jnp.eye(3) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
        R = manifold.project(A)
        assert manifold.belongs(R)

    def test_random_point_so4(self):
        """Test SO(n) generalizes beyond n=3."""
        manifold = SO(4)
        R = manifold.random_point(jax.random.PRNGKey(0))
        assert R.shape == (4, 4)
        assert manifold.belongs(R)


class TestSPDManifold:
    def test_random_point_is_spd(self):
        manifold = SPD(3)
        A = manifold.random_point(jax.random.PRNGKey(0))
        assert A.shape == (3, 3)
        assert manifold.belongs(A)

    def test_exp_log_roundtrip(self):
        manifold = SPD(3)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        A = manifold.random_point(k1)
        B = manifold.random_point(k2)
        V = manifold.log_map(A, B)
        B_recovered = manifold.exp_map(A, V)
        np.testing.assert_allclose(B_recovered, B, atol=1e-2)

    def test_distance_self_is_zero(self):
        manifold = SPD(3)
        A = manifold.random_point(jax.random.PRNGKey(0))
        d = manifold.distance(A, A)
        assert float(d) < 1e-4

    def test_distance_symmetric(self):
        manifold = SPD(3)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        A = manifold.random_point(k1)
        B = manifold.random_point(k2)
        np.testing.assert_allclose(
            manifold.distance(A, B), manifold.distance(B, A), atol=1e-3
        )

    def test_project_to_spd(self):
        """Test projection to nearest SPD matrix."""
        manifold = SPD(3)
        A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
        P = manifold.project(A)
        assert manifold.belongs(P)

    def test_random_point_spd5(self):
        """Test SPD(n) generalizes beyond n=3."""
        manifold = SPD(5)
        A = manifold.random_point(jax.random.PRNGKey(0))
        assert A.shape == (5, 5)
        assert manifold.belongs(A)


class TestGrassmannianManifold:
    def test_random_point_orthonormal(self):
        manifold = Grassmannian(5, 2)
        Q = manifold.random_point(jax.random.PRNGKey(0))
        assert Q.shape == (5, 2)
        assert manifold.belongs(Q)

    def test_exp_log_roundtrip(self):
        """Verify Exp(Log(Q2)) recovers Q2 on the Grassmannian."""
        manifold = Grassmannian(5, 2)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        Q1 = manifold.random_point(k1)
        Q2 = manifold.random_point(k2)
        V = manifold.log_map(Q1, Q2)
        Q2_recovered = manifold.exp_map(Q1, V)
        # Grassmannian points represent subspaces — compare via distance
        d = manifold.distance(Q2, Q2_recovered)
        assert float(d) < 1e-2

    def test_distance_self_is_zero(self):
        manifold = Grassmannian(5, 2)
        Q = manifold.random_point(jax.random.PRNGKey(0))
        d = manifold.distance(Q, Q)
        assert float(d) < 1e-3

    def test_distance_symmetric(self):
        manifold = Grassmannian(5, 2)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        Q1 = manifold.random_point(k1)
        Q2 = manifold.random_point(k2)
        np.testing.assert_allclose(
            manifold.distance(Q1, Q2), manifold.distance(Q2, Q1), atol=1e-4
        )

    def test_project_to_grassmannian(self):
        """Test projection to nearest orthonormal basis."""
        manifold = Grassmannian(5, 2)
        A = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        Q = manifold.project(A)
        assert manifold.belongs(Q)


# ---------------------------------------------------------------------------
# RNG on SO(3)
# ---------------------------------------------------------------------------

class TestRNGonSO3:
    @pytest.fixture
    def so3_data(self):
        """Generate cluster of rotations near identity."""
        manifold = SO(3)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 30)

        def perturbed_rotation(k):
            # Small random skew-symmetric perturbation
            A = jax.random.normal(k, (3, 3)) * 0.3
            skew = (A - A.T) / 2.0
            return jax.scipy.linalg.expm(skew)

        rotations = jax.vmap(perturbed_rotation)(keys)
        return rotations, manifold

    def test_fit_converges(self, so3_data):
        X, manifold = so3_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=30, lr_init=0.3, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        assert model.loss_ is not None
        assert model.n_iter_ > 0

    def test_prototypes_on_manifold(self, so3_data):
        X, manifold = so3_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=30, lr_init=0.3, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        for k in range(3):
            assert manifold.belongs(model.prototypes_[k])

    def test_predict(self, so3_data):
        X, manifold = so3_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=30, lr_init=0.3, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (30,)
        assert set(np.unique(labels).tolist()).issubset({0, 1, 2})

    def test_transform(self, so3_data):
        X, manifold = so3_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=30, lr_init=0.3, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        dists = model.transform(X)
        assert dists.shape == (30, 3)
        assert jnp.all(dists >= 0)

    def test_energy_decreases(self, so3_data):
        X, manifold = so3_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=30, lr_init=0.3, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        # Energy should generally decrease (not strictly monotone but trend)
        assert model.loss_history_[-1] < model.loss_history_[0]


# ---------------------------------------------------------------------------
# RNG on SPD(3)
# ---------------------------------------------------------------------------

class TestRNGonSPD:
    @pytest.fixture
    def spd_data(self):
        manifold = SPD(3)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 20)
        data = jax.vmap(manifold.random_point)(keys)
        return data, manifold

    def test_fit_converges(self, spd_data):
        X, manifold = spd_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        assert model.loss_ is not None

    def test_prototypes_are_spd(self, spd_data):
        X, manifold = spd_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        for k in range(3):
            assert manifold.belongs(model.prototypes_[k])

    def test_predict(self, spd_data):
        X, manifold = spd_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (20,)

    def test_transform(self, spd_data):
        X, manifold = spd_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        dists = model.transform(X)
        assert dists.shape == (20, 3)
        assert jnp.all(dists >= 0)

    def test_energy_decreases(self, spd_data):
        X, manifold = spd_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        assert model.loss_history_[-1] < model.loss_history_[0]


# ---------------------------------------------------------------------------
# RNG on Grassmannian
# ---------------------------------------------------------------------------

class TestRNGonGrassmannian:
    @pytest.fixture
    def grass_data(self):
        manifold = Grassmannian(5, 2)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 20)
        data = jax.vmap(manifold.random_point)(keys)
        return data, manifold

    def test_fit_converges(self, grass_data):
        X, manifold = grass_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        assert model.loss_ is not None

    def test_prototypes_orthonormal(self, grass_data):
        X, manifold = grass_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        for k in range(3):
            assert manifold.belongs(model.prototypes_[k])

    def test_predict(self, grass_data):
        X, manifold = grass_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (20,)

    def test_transform(self, grass_data):
        X, manifold = grass_data
        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)
        dists = model.transform(X)
        assert dists.shape == (20, 3)
        assert jnp.all(dists >= 0)


# ---------------------------------------------------------------------------
# Save/Load
# ---------------------------------------------------------------------------

class TestRNGSaveLoad:
    def test_save_load_so3(self, tmp_path):
        manifold = SO(3)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 20)

        def make_rot(k):
            A = jax.random.normal(k, (3, 3)) * 0.3
            skew = (A - A.T) / 2.0
            return jax.scipy.linalg.expm(skew)

        X = jax.vmap(make_rot)(keys)

        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.3, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)

        path = str(tmp_path / "rng_so3.npz")
        model.save(path)
        loaded = RiemannianNeuralGas.load(path)

        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )

    def test_save_load_spd(self, tmp_path):
        manifold = SPD(3)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 20)
        X = jax.vmap(manifold.random_point)(keys)

        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)

        path = str(tmp_path / "rng_spd.npz")
        model.save(path)
        loaded = RiemannianNeuralGas.load(path)

        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )

    def test_save_load_grassmannian(self, tmp_path):
        manifold = Grassmannian(5, 2)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 20)
        X = jax.vmap(manifold.random_point)(keys)

        model = RiemannianNeuralGas(
            manifold=manifold, n_prototypes=3,
            max_iter=20, lr_init=0.2, lr_final=0.01,
            random_seed=42,
        )
        model.fit(X)

        path = str(tmp_path / "rng_grass.npz")
        model.save(path)
        loaded = RiemannianNeuralGas.load(path)

        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )

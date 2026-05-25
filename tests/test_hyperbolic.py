"""Tests for Hyperbolic Poincare Ball manifold and Riemannian LVQ integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.core.manifolds import HyperbolicPoincare, _mobius_add, _conformal_factor
from prosemble.models import RiemannianSRNG, RiemannianSMNG


# ---------------------------------------------------------------------------
# Manifold axiom tests
# ---------------------------------------------------------------------------

class TestHyperbolicPoincareManifold:
    """Test that HyperbolicPoincare satisfies manifold axioms."""

    @pytest.fixture
    def manifold(self):
        return HyperbolicPoincare(4)

    @pytest.fixture
    def points(self, manifold):
        """Generate random points on the Poincare ball."""
        keys = jax.random.split(jax.random.PRNGKey(42), 5)
        return jnp.stack([manifold.random_point(k) for k in keys])

    def test_random_point_in_ball(self, manifold):
        """Random points must lie strictly inside the unit ball."""
        keys = jax.random.split(jax.random.PRNGKey(0), 20)
        for key in keys:
            p = manifold.random_point(key)
            assert p.shape == (4,)
            assert float(jnp.linalg.norm(p)) < 1.0

    def test_belongs(self, manifold, points):
        """All generated points must pass the belongs check."""
        for i in range(points.shape[0]):
            assert manifold.belongs(points[i])

    def test_belongs_rejects_outside(self, manifold):
        """Points outside the ball must fail belongs."""
        outside = jnp.array([1.5, 0.0, 0.0, 0.0])
        assert not manifold.belongs(outside)

    def test_project_clamps_to_ball(self, manifold):
        """Projection must bring points inside the ball."""
        outside = jnp.array([2.0, 1.0, 0.5, 0.3])
        projected = manifold.project(outside)
        assert float(jnp.linalg.norm(projected)) < 1.0
        assert manifold.belongs(projected)

    def test_project_preserves_interior(self, manifold, points):
        """Projection of interior points should not change them."""
        for i in range(points.shape[0]):
            projected = manifold.project(points[i])
            np.testing.assert_allclose(
                np.asarray(projected), np.asarray(points[i]), atol=1e-6
            )

    def test_distance_non_negative(self, manifold, points):
        """Geodesic distance must be non-negative."""
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                d = manifold.distance(points[i], points[j])
                assert float(d) >= 0.0

    def test_distance_self_zero(self, manifold, points):
        """Distance from a point to itself must be approximately zero."""
        for i in range(points.shape[0]):
            d = manifold.distance(points[i], points[i])
            # Limited by arcosh clamp (1e-6) for gradient stability
            assert float(d) < 2e-3

    def test_distance_symmetric(self, manifold, points):
        """Geodesic distance must be symmetric."""
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                d_ij = manifold.distance(points[i], points[j])
                d_ji = manifold.distance(points[j], points[i])
                np.testing.assert_allclose(float(d_ij), float(d_ji), atol=1e-5)

    def test_triangle_inequality(self, manifold, points):
        """Geodesic distance must satisfy the triangle inequality."""
        for i in range(3):
            for j in range(i + 1, 4):
                for k in range(j + 1, 5):
                    d_ij = float(manifold.distance(points[i], points[j]))
                    d_jk = float(manifold.distance(points[j], points[k]))
                    d_ik = float(manifold.distance(points[i], points[k]))
                    assert d_ik <= d_ij + d_jk + 1e-5

    def test_exp_log_roundtrip(self, manifold, points):
        """exp_x(log_x(y)) should approximately recover y."""
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                if i == j:
                    continue
                x, y = points[i], points[j]
                v = manifold.log_map(x, y)
                y_recovered = manifold.exp_map(x, v)
                # Float32 precision limits roundtrip accuracy for distant points
                np.testing.assert_allclose(
                    np.asarray(y_recovered), np.asarray(y), atol=5e-3
                )

    def test_log_exp_roundtrip(self, manifold, points):
        """log_x(exp_x(v)) should approximately recover v for small v."""
        x = points[0]
        # Use a small tangent vector
        v = jnp.array([0.05, -0.03, 0.02, -0.01])
        y = manifold.exp_map(x, v)
        v_recovered = manifold.log_map(x, y)
        np.testing.assert_allclose(
            np.asarray(v_recovered), np.asarray(v), atol=1e-3
        )

    def test_distance_squared_consistent(self, manifold, points):
        """distance_squared should equal distance^2."""
        for i in range(3):
            for j in range(i + 1, 4):
                d = manifold.distance(points[i], points[j])
                d2 = manifold.distance_squared(points[i], points[j])
                np.testing.assert_allclose(float(d2), float(d) ** 2, atol=1e-5)

    def test_injectivity_radius(self, manifold):
        """Poincare ball has infinite injectivity radius."""
        x = jnp.zeros(4)
        assert float(manifold.injectivity_radius(x)) == float('inf')


# ---------------------------------------------------------------------------
# Mobius addition tests
# ---------------------------------------------------------------------------

class TestMobiusAddition:

    def test_identity_element(self):
        """Mobius addition with origin should be identity."""
        x = jnp.array([0.3, -0.2, 0.1])
        zero = jnp.zeros(3)
        result = _mobius_add(zero, x)
        np.testing.assert_allclose(np.asarray(result), np.asarray(x), atol=1e-6)

    def test_inverse(self):
        """x + (-x) should give (approximately) the origin."""
        x = jnp.array([0.3, -0.2, 0.1])
        result = _mobius_add(x, -x)
        np.testing.assert_allclose(np.asarray(result), np.zeros(3), atol=1e-5)

    def test_conformal_factor_at_origin(self):
        """Conformal factor at origin should be 2."""
        origin = jnp.zeros(4)
        lam = _conformal_factor(origin)
        np.testing.assert_allclose(float(lam), 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Supervised Riemannian model integration tests
# ---------------------------------------------------------------------------

def _make_hyperbolic_data(d=4, n_per_class=15, seed=0):
    """Create two classes of points in the Poincare ball.

    Class 0 points cluster near one region, class 1 near another.
    """
    manifold = HyperbolicPoincare(d)
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # Class 0: points near a center
    center0 = jnp.zeros(d).at[0].set(0.3)
    noise0 = jax.random.normal(k1, (n_per_class, d)) * 0.05
    class0 = jax.vmap(manifold.project)(center0 + noise0)

    # Class 1: points near a different center
    center1 = jnp.zeros(d).at[0].set(-0.3)
    noise1 = jax.random.normal(k2, (n_per_class, d)) * 0.05
    class1 = jax.vmap(manifold.project)(center1 + noise1)

    X = jnp.concatenate([class0, class1], axis=0)
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    return manifold, X, y


class TestRiemannianSRNGHyperbolic:

    def test_fit_loss_decreases(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=20, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    def test_prototypes_on_manifold(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        n_protos = model.prototypes_.shape[0]
        protos = model.prototypes_.reshape(n_protos, *manifold.point_shape)
        for i in range(n_protos):
            assert manifold.belongs(protos[i]), f"Prototype {i} not on manifold"

    def test_predict_returns_valid_labels(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        unique_preds = set(np.asarray(preds).tolist())
        assert unique_preds.issubset({0, 1})

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = RiemannianSRNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_classification_accuracy(self):
        """With well-separated classes, accuracy should be reasonable."""
        manifold, X, y = _make_hyperbolic_data(d=4, n_per_class=20, seed=1)
        model = RiemannianSRNG(
            manifold=manifold, n_prototypes_per_class=2,
            max_iter=50, lr=0.01, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.7, f"Accuracy too low: {accuracy}"


class TestRiemannianSMNGHyperbolic:

    def test_fit_loss_decreases(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.001, use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[0] > model.loss_history_[-1]

    def test_predict_returns_valid_labels(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.001, use_scan=False,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        unique_preds = set(np.asarray(preds).tolist())
        assert unique_preds.issubset({0, 1})

    def test_save_load_roundtrip(self):
        manifold, X, y = _make_hyperbolic_data()
        model = RiemannianSMNG(
            manifold=manifold, n_prototypes_per_class=1,
            max_iter=10, lr=0.001, use_scan=False,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.npz')
            model.save(path)
            loaded = RiemannianSMNG.load(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

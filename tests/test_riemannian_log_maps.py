"""Exact log maps on SO(n) and the Grassmannian, and the ``log_map`` option."""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from prosemble.core.manifolds import SO, SPD, Grassmannian
from prosemble.models import (
    RiemannianSRNG, RiemannianSMNG, RiemannianSLNG, RiemannianSTNG,
    RiemannianDKGMLVQ, RiemannianDKGRLVQ,
)
from prosemble.models import riemannian_srng as rs
from prosemble.models.riemannian_srng import (
    _angle_ratio, _angle_ratio_prime, _angle_ratio_matrix,
    _so_log_map_diff, _so_log_map_exact,
    _grassmannian_log_map_diff, _grassmannian_log_map_exact,
)


def _rotation_about_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)


def _so_pairs(n, count=25, min_gap=0.1):
    """Random pairs on SO(n) that stay ``min_gap`` away from the cut locus."""
    so = SO(n)
    pairs = []
    seed = 0
    while len(pairs) < count:
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        seed += 1
        R, S = so.random_point(k1), so.random_point(k2)
        eigvals = np.linalg.eigvals(np.asarray(R, np.float64).T @ np.asarray(S, np.float64))
        if np.abs(eigvals + 1.0).min() > min_gap:
            pairs.append((R, S))
    return so, pairs


# ---------------------------------------------------------------------------
# The scalar function behind both log maps
# ---------------------------------------------------------------------------

class TestAngleRatio:

    def test_matches_definition(self):
        v = np.array([1e-4, 0.01, 0.3, 0.7, 0.979, 0.981, 0.999, 1.0 - 1e-6])
        theta = np.arccos(np.sqrt(v))
        expected = theta / (np.sin(theta) * np.cos(theta))
        np.testing.assert_allclose(_angle_ratio(jnp.asarray(v, jnp.float32)), expected, rtol=2e-5)

    def test_value_at_identity(self):
        assert float(_angle_ratio(jnp.float32(1.0))) == pytest.approx(1.0)

    def test_series_and_closed_form_join(self):
        below, above = rs._SERIES_COS2 - 1e-4, rs._SERIES_COS2 + 1e-4
        gap = float(_angle_ratio(jnp.float32(above)) - _angle_ratio(jnp.float32(below)))
        assert abs(gap) < 5e-4

    def test_is_clamped_at_the_cut_locus(self):
        at_clamp = float(_angle_ratio(jnp.float32(rs._CUT_LOCUS_COS2)))
        assert float(_angle_ratio(jnp.float32(0.0))) == pytest.approx(at_clamp)
        assert np.isfinite(at_clamp)
        assert float(_angle_ratio_prime(jnp.float32(0.0))) == 0.0

    def test_derivative(self):
        v = np.array([0.01, 0.3, 0.7, 0.95, 0.99, 0.999])
        h = 1e-6

        def phi(x):
            theta = np.arccos(np.sqrt(x))
            return theta / (np.sin(theta) * np.cos(theta))

        expected = (phi(v + h) - phi(v - h)) / (2 * h)
        np.testing.assert_allclose(
            _angle_ratio_prime(jnp.asarray(v, jnp.float32)), expected, rtol=1e-3)

    def test_matrix_function_has_finite_gradient_for_repeated_eigenvalues(self):
        for M in (jnp.eye(3), 0.5 * jnp.eye(3), jnp.diag(jnp.array([0.4, 0.4, 1.0]))):
            grad = jax.grad(lambda m: jnp.sum(_angle_ratio_matrix(m) ** 2))(M)
            assert bool(jnp.all(jnp.isfinite(grad)))


# ---------------------------------------------------------------------------
# SO(n)
# ---------------------------------------------------------------------------

class TestSOLogMap:

    @pytest.mark.parametrize('n', [2, 3, 4, 5])
    def test_matches_matrix_logarithm(self, n):
        so, pairs = _so_pairs(n)
        for R, S in pairs:
            R64, S64 = np.asarray(R, np.float64), np.asarray(S, np.float64)
            expected = R64 @ scipy.linalg.logm(R64.T @ S64).real
            np.testing.assert_allclose(_so_log_map_exact(R, S), expected, atol=2e-3)

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_exp_map_inverts_it(self, n):
        so, pairs = _so_pairs(n, count=10)
        for R, S in pairs:
            np.testing.assert_allclose(so.exp_map(R, _so_log_map_exact(R, S)), S, atol=2e-3)

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_length_is_the_geodesic_distance(self, n):
        # The metric is <U, V> = tr(U^T V) / 2, so |Log|_F = sqrt(2) d.
        so, pairs = _so_pairs(n, count=10)
        for R, S in pairs:
            np.testing.assert_allclose(
                jnp.linalg.norm(_so_log_map_exact(R, S)),
                jnp.sqrt(2.0) * so.distance(R, S), rtol=2e-3)

    def test_closed_form_on_so3_agrees_with_eigenvalue_path(self):
        so, pairs = _so_pairs(3)
        for R, S in pairs:
            Q = R.T @ S
            K = (Q - Q.T) / 2.0
            G = _angle_ratio_matrix((2.0 * jnp.eye(3) + Q + Q.T) / 4.0)
            general = R @ ((K @ G + G @ K) / 2.0)
            np.testing.assert_allclose(_so_log_map_exact(R, S), general, atol=2e-4)

    def test_length_grows_with_the_angle_where_first_order_does_not(self):
        base = _rotation_about_z(0.0)
        angles = np.radians([30.0, 90.0, 150.0, 175.0])
        exact = [float(jnp.linalg.norm(_so_log_map_exact(base, _rotation_about_z(a))))
                 for a in angles]
        first = [float(jnp.linalg.norm(_so_log_map_diff(base, _rotation_about_z(a))))
                 for a in angles]
        np.testing.assert_allclose(exact, np.sqrt(2.0) * angles, rtol=1e-3)
        assert exact == sorted(exact)
        # First order: sqrt(2) sin(angle), which shrinks again beyond 90 degrees.
        assert first[3] < first[2] < first[1]
        assert first[3] < first[0]

    def test_agrees_with_first_order_for_nearby_rotations(self):
        base = SO(3).random_point(jax.random.PRNGKey(0))
        near = base @ _rotation_about_z(0.01)
        np.testing.assert_allclose(
            _so_log_map_exact(base, near), _so_log_map_diff(base, near), atol=1e-5)

    @pytest.mark.parametrize('n', [2, 3, 4])
    def test_gradient_is_finite(self, n):
        so, pairs = _so_pairs(n, count=3)

        def length_squared(R, S):
            return jnp.sum(_so_log_map_exact(R, S) ** 2)

        for R, S in pairs:
            for args in ((R, S), (R, R)):
                grads = jax.grad(length_squared, argnums=(0, 1))(*args)
                assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    @pytest.mark.parametrize('n', [3, 4])
    def test_gradient_matches_finite_differences(self, n):
        so, pairs = _so_pairs(n, count=3, min_gap=0.5)
        weights = jnp.arange(float(n * n)).reshape(n, n) / (n * n)

        def f(R, S):
            return jnp.sum(jnp.sin(_so_log_map_exact(R, S)) * weights)

        for R, S in pairs:
            direction = jax.random.normal(jax.random.PRNGKey(7), (n, n))
            h = 1e-2
            numeric = (f(R, S + h * direction) - f(R, S - h * direction)) / (2 * h)
            analytic = jnp.sum(jax.grad(f, argnums=1)(R, S) * direction)
            np.testing.assert_allclose(analytic, numeric, rtol=5e-2, atol=5e-3)

    def test_is_finite_and_small_at_the_cut_locus(self):
        base = _rotation_about_z(0.0)
        at_pi = _so_log_map_exact(base, _rotation_about_z(np.pi))
        assert bool(jnp.all(jnp.isfinite(at_pi)))
        assert float(jnp.linalg.norm(at_pi)) < 0.1
        # Just inside, the length is the full angle.
        inside = _so_log_map_exact(base, _rotation_about_z(np.pi - 0.02))
        assert float(jnp.linalg.norm(inside)) == pytest.approx(
            np.sqrt(2.0) * (np.pi - 0.02), rel=1e-2)


# ---------------------------------------------------------------------------
# Grassmannian
# ---------------------------------------------------------------------------

def _grassmannian_pairs(n, k, count=15, max_angle=1.45):
    gr = Grassmannian(n, k)
    pairs = []
    seed = 0
    while len(pairs) < count:
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        seed += 1
        Q1, Q2 = gr.random_point(k1), gr.random_point(k2)
        cosines = np.linalg.svd(np.asarray(Q1.T @ Q2, np.float64), compute_uv=False)
        if np.arccos(np.clip(cosines, -1, 1)).max() < max_angle:
            pairs.append((Q1, Q2))
    return gr, pairs


class TestGrassmannianLogMap:

    @pytest.mark.parametrize('n, k', [(4, 1), (4, 2), (6, 2), (7, 3)])
    def test_exp_map_inverts_it(self, n, k):
        gr, pairs = _grassmannian_pairs(n, k)
        for Q1, Q2 in pairs:
            back = gr.exp_map(Q1, _grassmannian_log_map_exact(Q1, Q2))
            np.testing.assert_allclose(gr.distance(back, Q2), 0.0, atol=5e-3)

    @pytest.mark.parametrize('n, k', [(4, 1), (4, 2), (6, 2), (7, 3)])
    def test_length_is_the_geodesic_distance(self, n, k):
        gr, pairs = _grassmannian_pairs(n, k)
        for Q1, Q2 in pairs:
            np.testing.assert_allclose(
                jnp.linalg.norm(_grassmannian_log_map_exact(Q1, Q2)),
                gr.distance(Q1, Q2), rtol=5e-3)

    def test_is_horizontal(self):
        gr, pairs = _grassmannian_pairs(6, 2)
        for Q1, Q2 in pairs:
            np.testing.assert_allclose(
                Q1.T @ _grassmannian_log_map_exact(Q1, Q2), 0.0, atol=1e-5)

    def test_does_not_depend_on_the_basis_of_the_target(self):
        """Another basis of the same subspace is the same point."""
        gr, pairs = _grassmannian_pairs(6, 2)
        for i, (Q1, Q2) in enumerate(pairs):
            rotation = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(i), (2, 2)))[0]
            rebased = Q2 @ rotation
            np.testing.assert_allclose(
                _grassmannian_log_map_exact(Q1, rebased),
                _grassmannian_log_map_exact(Q1, Q2), atol=2e-3)
            # The first-order form changes with the basis.
            change = jnp.abs(_grassmannian_log_map_diff(Q1, rebased)
                             - _grassmannian_log_map_diff(Q1, Q2)).max()
            assert float(change) > 1e-2

    def test_gradient_is_finite(self):
        gr, pairs = _grassmannian_pairs(5, 2, count=3)

        def length_squared(Q1, Q2):
            return jnp.sum(_grassmannian_log_map_exact(Q1, Q2) ** 2)

        for Q1, Q2 in pairs:
            for args in ((Q1, Q2), (Q1, Q1)):
                grads = jax.grad(length_squared, argnums=(0, 1))(*args)
                assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    def test_is_finite_at_the_cut_locus(self):
        Q1 = jnp.eye(4)[:, :2]
        Q2 = jnp.eye(4)[:, 2:]          # orthogonal subspace
        tangent = _grassmannian_log_map_exact(Q1, Q2)
        assert bool(jnp.all(jnp.isfinite(tangent)))


# ---------------------------------------------------------------------------
# Model option
# ---------------------------------------------------------------------------

def _half_turn_problem(n_per_class=30, seed=0):
    """Two classes of rotations about one axis, each spanning 160 degrees.

    Class 0 is centred on the identity and class 1 on the half turn. For
    every sample the angles to the two centres add up to 180 degrees, so
    a tangent length proportional to sin(angle) cannot tell them apart.
    """
    rng = np.random.default_rng(seed)
    spread = np.radians(80.0)
    angles = np.concatenate([
        rng.uniform(-spread, spread, n_per_class),
        np.pi + rng.uniform(-spread, spread, n_per_class),
    ])
    X = jnp.stack([_rotation_about_z(a) for a in angles]).reshape(len(angles), -1)
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    centres = jnp.stack([_rotation_about_z(0.0), _rotation_about_z(np.pi)]).reshape(2, -1)
    return X, y, centres


LOG_MAP_MODELS = [RiemannianSMNG, RiemannianSLNG, RiemannianSTNG,
                  RiemannianDKGMLVQ, RiemannianDKGRLVQ]


class TestLogMapOption:

    @pytest.mark.parametrize('cls', LOG_MAP_MODELS)
    def test_default_depends_on_the_manifold(self, cls):
        assert cls(manifold=SO(3)).log_map == 'exact'
        assert cls(manifold=SO(4)).log_map == 'exact'
        assert cls(manifold=Grassmannian(4, 2)).log_map == 'first_order'
        assert cls(manifold=Grassmannian(4, 2), log_map='exact').log_map == 'exact'

    @pytest.mark.parametrize('cls', LOG_MAP_MODELS)
    def test_invalid_value(self, cls):
        with pytest.raises(ValueError, match='log_map'):
            cls(manifold=SO(3), log_map='second_order')

    def test_distance_only_models_do_not_take_it(self):
        with pytest.raises(TypeError):
            RiemannianSRNG(manifold=SO(3), log_map='exact')

    @pytest.mark.parametrize('cls', [RiemannianSMNG, RiemannianSLNG])
    def test_exact_separates_classes_first_order_cannot(self, cls):
        X, y, centres = _half_turn_problem()
        accuracy = {}
        for log_map in ('exact', 'first_order'):
            model = cls(manifold=SO(3), n_prototypes_per_class=1, max_iter=2,
                        lr=1e-4, log_map=log_map)
            model.fit(X, y, initial_prototypes=centres, initial_labels=jnp.array([0, 1]))
            accuracy[log_map] = float(jnp.mean(model.predict(X) == y))
        assert accuracy['exact'] >= 0.95
        assert accuracy['first_order'] <= 0.75

    @pytest.mark.parametrize('cls', LOG_MAP_MODELS)
    @pytest.mark.parametrize('manifold', [SO(3), Grassmannian(4, 2), SPD(3)],
                             ids=['SO3', 'Gr42', 'SPD3'])
    def test_training_stays_finite(self, cls, manifold):
        keys = jax.random.split(jax.random.PRNGKey(0), 24)
        X = jnp.stack([manifold.random_point(k) for k in keys]).reshape(24, -1)
        y = jnp.array([0, 1] * 12)
        model = cls(manifold=manifold, n_prototypes_per_class=2, max_iter=15, lr=0.02,
                    log_map='exact')
        model.fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.prototypes_)))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(model.loss_history_))))

    @pytest.mark.parametrize('cls', [RiemannianSMNG, RiemannianSTNG, RiemannianDKGMLVQ])
    def test_save_load_keeps_log_map(self, cls):
        so = SO(3)
        keys = jax.random.split(jax.random.PRNGKey(1), 20)
        X = jnp.stack([so.random_point(k) for k in keys]).reshape(20, -1)
        y = jnp.array([0, 1] * 10)
        for log_map in ('exact', 'first_order'):
            model = cls(manifold=so, max_iter=5, log_map=log_map)
            model.fit(X, y)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'model.npz')
                model.save(path)
                loaded = cls.load(path)
            assert loaded.log_map == log_map
            np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    def test_files_without_log_map_load_as_first_order(self):
        so = SO(3)
        keys = jax.random.split(jax.random.PRNGKey(2), 20)
        X = jnp.stack([so.random_point(k) for k in keys]).reshape(20, -1)
        y = jnp.array([0, 1] * 10)
        model = RiemannianSMNG(manifold=so, max_iter=5, log_map='first_order')
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            data = dict(np.load(path, allow_pickle=False))
            metadata = json.loads(str(data['__metadata__']))
            del metadata['hyperparams']['log_map']
            data['__metadata__'] = np.array(json.dumps(metadata))
            legacy = os.path.join(tmp, 'legacy.npz')
            np.savez_compressed(legacy, **data)
            loaded = RiemannianSMNG.load(legacy)
        assert loaded.log_map == 'first_order'
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

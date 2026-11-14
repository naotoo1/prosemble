"""Intrinsic (Riemannian) prototype updates on manifolds.

Covers the tangent-space geometry added to the manifolds, the optax
wrapper that turns an optimizer step into a geodesic step, and the
``update_rule`` option of the supervised Riemannian models.
"""

import json
import os
import tempfile
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.core.riemannian_optim import (
    riemannian_update, supports_riemannian_update,
)
from prosemble.models import RiemannianSRNG, RiemannianSMNG, RiemannianSTNG
from prosemble.models.riemannian_srng import (
    _so_chordal_distance_squared,
    _spd_distance_squared_diff,
    _grassmannian_distance_squared_diff,
    _hyperbolic_distance_squared_diff,
)


def _hyperbolic_d2(x, w):
    return _hyperbolic_distance_squared_diff(x, w, eps=1e-5)


MANIFOLDS = [
    pytest.param(SO(3), _so_chordal_distance_squared, id='SO3'),
    pytest.param(SO(4), _so_chordal_distance_squared, id='SO4'),
    pytest.param(SPD(3), _spd_distance_squared_diff, id='SPD3'),
    pytest.param(Grassmannian(5, 2), _grassmannian_distance_squared_diff, id='Gr52'),
    pytest.param(HyperbolicPoincare(3), _hyperbolic_d2, id='Poincare3'),
]


def _points(manifold, seed=0):
    """Two points, valid frame coordinates and the matching tangent vector."""
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    x, w = manifold.random_point(k[0]), manifold.random_point(k[1])
    if isinstance(manifold, HyperbolicPoincare):
        # Stay away from the boundary, where the distance is regularised.
        x, w = 0.5 * x, 0.5 * w
    coords = manifold.gradient_to_frame(w, jax.random.normal(k[2], w.shape))
    coords = coords / jnp.linalg.norm(coords)
    return x, w, coords, manifold.frame_to_tangent(w, coords)


# ---------------------------------------------------------------------------
# Tangent-space geometry of the manifolds
# ---------------------------------------------------------------------------

class TestTangentGeometry:

    @pytest.mark.parametrize('manifold, d2', MANIFOLDS)
    def test_frame_is_orthonormal(self, manifold, d2):
        """Riemannian norm of a tangent vector = Euclidean norm of its coordinates."""
        _, w, coords, v = _points(manifold)
        np.testing.assert_allclose(
            manifold.tangent_norm(w, v), jnp.linalg.norm(coords), rtol=1e-4)

    @pytest.mark.parametrize('manifold, d2', MANIFOLDS)
    def test_geodesic_has_unit_speed(self, manifold, d2):
        """d(w, Exp_w(t v)) = t |v|_w inside the injectivity radius."""
        _, w, _, v = _points(manifold)
        t = 0.3
        moved = manifold.exp_map(w, t * v)
        np.testing.assert_allclose(
            manifold.distance(w, moved), t * manifold.tangent_norm(w, v), rtol=2e-2)

    @pytest.mark.parametrize('manifold, d2', MANIFOLDS)
    def test_exp_map_stays_on_manifold(self, manifold, d2):
        _, w, _, v = _points(manifold)
        assert bool(manifold.belongs(manifold.exp_map(w, 0.5 * v)))

    @pytest.mark.parametrize('manifold, d2', MANIFOLDS)
    def test_exp_map_of_zero_is_identity(self, manifold, d2):
        _, w, _, v = _points(manifold)
        np.testing.assert_allclose(manifold.exp_map(w, 0.0 * v), w, atol=2e-6)

    @pytest.mark.parametrize('manifold, d2', MANIFOLDS)
    def test_exp_map_is_first_order_consistent(self, manifold, d2):
        """Exp_w(h v) = w + h v + O(h^2) as arrays, not only as points."""
        _, w, _, v = _points(manifold)
        h = 1e-2
        err = jnp.linalg.norm(manifold.exp_map(w, h * v) - w - h * v)
        assert float(err) < 0.1 * h * float(jnp.linalg.norm(v))

    @pytest.mark.parametrize('manifold, d2', MANIFOLDS)
    def test_frame_gradient_is_the_riemannian_gradient(self, manifold, d2):
        """d/dt f(Exp_w(t v)) at t = 0 equals <grad coordinates, v coordinates>."""
        x, w, coords, v = _points(manifold)

        def f(p):
            return d2(x, p)

        grad_coords = manifold.gradient_to_frame(w, jax.grad(f)(w))
        h = 1e-2
        directional = (f(manifold.exp_map(w, h * v))
                       - f(manifold.exp_map(w, -h * v))) / (2 * h)
        np.testing.assert_allclose(
            jnp.sum(grad_coords * coords), directional, rtol=2e-2, atol=2e-3)


class TestGrassmannianRepresentative:

    def test_project_is_identity_on_orthonormal_input(self):
        gr = Grassmannian(6, 3)
        for seed in range(5):
            Q = gr.random_point(jax.random.PRNGKey(seed))
            np.testing.assert_allclose(gr.project(Q), Q, atol=1e-5)

    def test_project_keeps_the_subspace(self):
        gr = Grassmannian(6, 3)
        A = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
        Q = gr.project(A)
        assert bool(gr.belongs(Q))
        # Same column space: projector onto span(Q) reproduces A.
        np.testing.assert_allclose(Q @ (Q.T @ A), A, atol=1e-4)

    def test_exp_map_inverts_log_map(self):
        gr = Grassmannian(5, 2)
        k1, k2 = jax.random.split(jax.random.PRNGKey(2))
        Q1, Q2 = gr.random_point(k1), gr.random_point(k2)
        back = gr.exp_map(Q1, gr.log_map(Q1, Q2))
        np.testing.assert_allclose(gr.distance(back, Q2), 0.0, atol=2e-3)


class TestHyperbolicIsUnbiased:

    def test_mobius_addition_of_zero(self):
        from prosemble.core.manifolds import _mobius_add
        x = jnp.array([0.6, -0.3, 0.2])
        np.testing.assert_allclose(_mobius_add(x, jnp.zeros(3), eps=1e-5), x, atol=1e-7)

    def test_log_map_inverts_exp_map(self):
        hyp = HyperbolicPoincare(3)
        x = jnp.array([0.4, 0.1, -0.3])
        v = jnp.array([0.2, -0.1, 0.05])
        np.testing.assert_allclose(hyp.log_map(x, hyp.exp_map(x, v)), v, atol=1e-5)


# ---------------------------------------------------------------------------
# Optimizer wrapper
# ---------------------------------------------------------------------------

class TestRiemannianUpdate:

    def test_sgd_on_spd_is_riemannian_gradient_descent(self):
        spd = SPD(3)
        W = spd.random_point(jax.random.PRNGKey(0))
        G = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
        lr = 0.05
        opt = riemannian_update(optax.sgd(lr), spd)
        params = {'prototypes': W.reshape(1, -1)}
        updates, _ = opt.update({'prototypes': G.reshape(1, -1)}, opt.init(params), params)
        new = optax.apply_updates(params, updates)['prototypes'].reshape(3, 3)

        # Affine-invariant metric: grad f = W sym(G) W.
        rgrad = W @ ((G + G.T) / 2) @ W
        np.testing.assert_allclose(new, spd.exp_map(W, -lr * rgrad), rtol=1e-4, atol=1e-5)

    def test_sgd_on_so_is_riemannian_gradient_descent(self):
        so = SO(3)
        R = so.random_point(jax.random.PRNGKey(0))
        G = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
        lr = 0.05
        opt = riemannian_update(optax.sgd(lr), so)
        params = {'prototypes': R.reshape(1, -1)}
        updates, _ = opt.update({'prototypes': G.reshape(1, -1)}, opt.init(params), params)
        new = optax.apply_updates(params, updates)['prototypes'].reshape(3, 3)

        # Metric <U, V> = tr(U^T V) / 2: grad f = R (R^T G - G^T R).
        rgrad = R @ (R.T @ G - G.T @ R)
        np.testing.assert_allclose(new, so.exp_map(R, -lr * rgrad), atol=1e-5)

    @pytest.mark.parametrize('tau', [0.25, 0.9])
    def test_step_is_limited_to_injectivity_radius(self, tau):
        so = SO(3)
        R = so.random_point(jax.random.PRNGKey(0))
        G = 1e3 * jax.random.normal(jax.random.PRNGKey(1), (3, 3))
        opt = riemannian_update(optax.sgd(1.0), so, tau=tau)
        params = {'prototypes': R.reshape(1, -1)}
        updates, _ = opt.update({'prototypes': G.reshape(1, -1)}, opt.init(params), params)
        new = optax.apply_updates(params, updates)['prototypes'].reshape(3, 3)
        np.testing.assert_allclose(so.distance(R, new), tau * jnp.pi, rtol=1e-3)

    def test_zero_gradient_gives_zero_update(self):
        for manifold in (SO(3), SPD(3), Grassmannian(5, 2), HyperbolicPoincare(3)):
            w = manifold.random_point(jax.random.PRNGKey(0))
            opt = riemannian_update(optax.adam(0.1), manifold)
            params = {'prototypes': w.reshape(1, -1)}
            updates, _ = opt.update(
                {'prototypes': jnp.zeros_like(params['prototypes'])},
                opt.init(params), params)
            np.testing.assert_allclose(updates['prototypes'], 0.0, atol=5e-6)

    def test_other_parameters_are_left_to_the_optimizer(self):
        spd = SPD(3)
        params = {'prototypes': spd.random_point(jax.random.PRNGKey(0)).reshape(1, -1),
                  'omega': jnp.eye(3)}
        grads = {'prototypes': jnp.ones((1, 9)), 'omega': jnp.full((3, 3), 0.5)}
        plain = optax.sgd(0.1)
        wrapped = riemannian_update(plain, spd)
        expected, _ = plain.update(grads, plain.init(params), params)
        got, _ = wrapped.update(grads, wrapped.init(params), params)
        np.testing.assert_allclose(got['omega'], expected['omega'])

    def test_masked_prototypes_pass_through(self):
        spd = SPD(3)
        params = {'prototypes': spd.random_point(jax.random.PRNGKey(0)).reshape(1, -1),
                  'omega': jnp.eye(3)}
        grads = {'prototypes': jnp.ones((1, 9)), 'omega': jnp.ones((3, 3))}
        opt = optax.masked(riemannian_update(optax.sgd(0.1), spd),
                           {'prototypes': False, 'omega': True})
        updates, _ = opt.update(grads, opt.init(params), params)
        np.testing.assert_allclose(updates['omega'], -0.1 * jnp.ones((3, 3)))

    def test_requires_params(self):
        opt = riemannian_update(optax.sgd(0.1), SPD(3))
        params = {'prototypes': jnp.eye(3).reshape(1, -1)}
        with pytest.raises(ValueError, match='current parameters'):
            opt.update(params, opt.init(params))

    def test_supports_check(self):
        assert supports_riemannian_update(SPD(3))
        assert not supports_riemannian_update(object())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _data(manifold, n_per_class=12, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * n_per_class)
    points = jnp.stack([manifold.random_point(k) for k in keys])
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    return points.reshape(2 * n_per_class, -1), y


def _spd_scale_problem(n_per_class=20, seed=0):
    """Two SPD(3) classes that differ by a scale factor of 20."""
    spd = SPD(3)
    X, y = _data(spd, n_per_class, seed)
    X = X * jnp.where(y == 1, 20.0, 1.0)[:, None]
    return spd, X, y


MODEL_MANIFOLDS = [
    pytest.param(SO(3), id='SO3'),
    pytest.param(SPD(3), id='SPD3'),
    pytest.param(Grassmannian(4, 2), id='Gr42'),
    pytest.param(HyperbolicPoincare(3), id='Poincare3'),
]


class TestModelUpdateRule:

    def test_default_is_riemannian(self):
        assert RiemannianSRNG(manifold=SPD(3)).update_rule == 'riemannian'

    @pytest.mark.parametrize('manifold', MODEL_MANIFOLDS)
    @pytest.mark.parametrize('use_scan', [False, True])
    def test_prototypes_stay_on_manifold(self, manifold, use_scan):
        X, y = _data(manifold)
        model = RiemannianSRNG(manifold=manifold, n_prototypes_per_class=2,
                               max_iter=30, lr=0.05, use_scan=use_scan)
        model.fit(X, y)
        protos = model.prototypes_.reshape(-1, *manifold.point_shape)
        assert bool(jnp.all(jnp.isfinite(protos)))
        assert all(bool(manifold.belongs(p)) for p in protos)

    @pytest.mark.parametrize('cls', [RiemannianSRNG, RiemannianSMNG, RiemannianSTNG])
    def test_rules_differ(self, cls):
        spd = SPD(3)
        X, y = _data(spd)
        fitted = {}
        for rule in ('riemannian', 'projected'):
            model = cls(manifold=spd, max_iter=10, lr=0.05, update_rule=rule)
            model.fit(X, y)
            fitted[rule] = np.asarray(model.prototypes_)
        assert not np.allclose(fitted['riemannian'], fitted['projected'], atol=1e-4)

    def test_riemannian_sgd_learns_scale_problem(self):
        spd, X, y = _spd_scale_problem()
        model = RiemannianSRNG(manifold=spd, max_iter=60, lr=1.0, optimizer='sgd')
        model.fit(X, y)
        assert float(jnp.mean(model.predict(X) == y)) >= 0.9

    def test_spd_training_is_scale_equivariant(self):
        """Rescaling SPD data rescales the prototypes and nothing else.

        The affine-invariant distance does not change under X -> c X, so
        an intrinsic update must give prototypes c W. A Euclidean step
        does not have this property.
        """
        spd = SPD(3)
        X, y = _data(spd)
        c = 50.0

        def fit(data, rule):
            model = RiemannianSRNG(manifold=spd, n_prototypes_per_class=2,
                                   max_iter=25, lr=0.05, update_rule=rule)
            model.fit(data, y)
            return np.asarray(model.prototypes_)

        np.testing.assert_allclose(
            fit(c * X, 'riemannian'), c * fit(X, 'riemannian'), rtol=5e-3, atol=5e-3)
        assert not np.allclose(
            fit(c * X, 'projected'), c * fit(X, 'projected'), rtol=5e-2, atol=5e-2)

    def test_invalid_update_rule(self):
        with pytest.raises(ValueError, match='update_rule'):
            RiemannianSRNG(manifold=SPD(3), update_rule='euclidean')

    @pytest.mark.parametrize('tau', [0.0, -0.5, 1.5])
    def test_invalid_tau(self, tau):
        with pytest.raises(ValueError, match='tau'):
            RiemannianSRNG(manifold=SPD(3), tau=tau)

    def test_manifold_without_frame_falls_back(self):
        class Flat:
            """Euclidean space as a manifold, without the tangent-frame methods."""

            point_shape = (3,)

            def distance_squared(self, p, q):
                return jnp.sum((p - q) ** 2)

            def distance(self, p, q):
                return jnp.sqrt(self.distance_squared(p, q) + 1e-12)

            def log_map(self, base, target):
                return target - base

            def exp_map(self, base, tangent):
                return base + tangent

            def random_point(self, key):
                return jax.random.normal(key, (3,))

            def belongs(self, p):
                return jnp.all(jnp.isfinite(p))

            def project(self, p):
                return p

            def injectivity_radius(self, p):
                return jnp.inf

        flat = Flat()
        assert not supports_riemannian_update(flat)
        X, y = _data(flat)
        with pytest.warns(UserWarning, match="falling back to update_rule='projected'"):
            model = RiemannianSRNG(manifold=flat, max_iter=5)
        model.fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.prototypes_)))

    def test_gradient_accumulation(self):
        spd = SPD(3)
        X, y = _data(spd)
        model = RiemannianSRNG(manifold=spd, max_iter=12, lr=0.05,
                               gradient_accumulation_steps=3)
        model.fit(X, y)
        protos = model.prototypes_.reshape(-1, 3, 3)
        assert all(bool(spd.belongs(p)) for p in protos)


class TestUpdateRulePersistence:

    def test_save_load_keeps_update_rule(self):
        spd = SPD(3)
        X, y = _data(spd)
        for rule in ('riemannian', 'projected'):
            model = RiemannianSRNG(manifold=spd, max_iter=5, update_rule=rule)
            model.fit(X, y)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'model.npz')
                model.save(path)
                loaded = RiemannianSRNG.load(path)
            assert loaded.update_rule == rule
            np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    def test_files_without_update_rule_load_as_projected(self):
        spd = SPD(3)
        X, y = _data(spd)
        model = RiemannianSRNG(manifold=spd, max_iter=5)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            data = dict(np.load(path, allow_pickle=False))
            metadata = json.loads(str(data['__metadata__']))
            del metadata['hyperparams']['update_rule']
            data['__metadata__'] = np.array(json.dumps(metadata))
            legacy = os.path.join(tmp, 'legacy.npz')
            np.savez_compressed(legacy, **data)
            loaded = RiemannianSRNG.load(legacy)
        assert loaded.update_rule == 'projected'
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

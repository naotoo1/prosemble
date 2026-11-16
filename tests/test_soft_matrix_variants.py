"""Matrix and local matrix soft variants.

DKMRSLVQ, DKLMRSLVQ, WMRSLVQ, WLMRSLVQ and RiemannianMRSLVQ.
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from prosemble.core.distance import (
    wasserstein2_distance_matrix, wasserstein2_pushforward_distance_matrix,
)
from prosemble.core.kernel import (
    kernel_distance_squared_matrix, kernel_distance_squared_per_proto,
)
from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.models import (
    _MODEL_REGISTRY, DKRSLVQ, DKMRSLVQ, DKLMRSLVQ, WRSLVQ, WMRSLVQ, WLMRSLVQ,
    RiemannianRSLVQ, RiemannianMRSLVQ,
)
from prosemble.models.riemannian_srng import _tangent_coordinates
from prosemble.models.soft_mixin import SoftAssignmentMixin, SoftMetricMixin


@pytest.fixture(scope='module')
def iris():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y)


def _manifold_data(manifold, n_per_class=14, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * n_per_class)
    points = jnp.stack([manifold.random_point(k) for k in keys])
    y = jnp.array([0] * n_per_class + [1] * n_per_class)
    return points.reshape(2 * n_per_class, -1), y


def _random(seed, *shape):
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


# (model, the model without a matrix, local)
EUCLIDEAN = [
    pytest.param(DKMRSLVQ, DKRSLVQ, False, id='DKMRSLVQ'),
    pytest.param(DKLMRSLVQ, DKRSLVQ, True, id='DKLMRSLVQ'),
    pytest.param(WMRSLVQ, WRSLVQ, False, id='WMRSLVQ'),
    pytest.param(WLMRSLVQ, WRSLVQ, True, id='WLMRSLVQ'),
]
RIEMANNIAN = [
    pytest.param(RiemannianMRSLVQ, False, id='RiemannianMRSLVQ'),
]
MANIFOLDS = [
    pytest.param(SO(3), id='SO3'),
    pytest.param(SPD(3), id='SPD3'),
    pytest.param(Grassmannian(4, 2), id='Gr42'),
    pytest.param(HyperbolicPoincare(3), id='Poincare3'),
]


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

class TestMatrixKernelDistance:

    def _case(self, n=9, p=4, d=5):
        X, W = _random(0, n, d), _random(1, p, d)
        bandwidths = jnp.abs(_random(2, p)) + 0.5
        return X, W, bandwidths

    def test_identity_gives_the_plain_kernel_distance(self):
        X, W, bandwidths = self._case()
        np.testing.assert_allclose(
            kernel_distance_squared_matrix(X, W, bandwidths, jnp.eye(5)),
            kernel_distance_squared_per_proto(X, W, bandwidths), rtol=1e-5, atol=1e-6)

    def test_is_the_plain_kernel_distance_of_the_mapped_data(self):
        X, W, bandwidths = self._case()
        omega = _random(3, 5, 3)
        np.testing.assert_allclose(
            kernel_distance_squared_matrix(X, W, bandwidths, omega),
            kernel_distance_squared_per_proto(X @ omega, W @ omega, bandwidths),
            rtol=1e-4, atol=1e-5)

    def test_stays_bounded_for_a_large_matrix(self):
        X, W, bandwidths = self._case()
        distances = kernel_distance_squared_matrix(X, W, bandwidths, 1e3 * jnp.eye(5))
        assert bool(jnp.all(jnp.isfinite(distances)))
        assert float(distances.min()) >= 0.0 and float(distances.max()) <= 2.0

    def test_equal_local_matrices_give_the_global_distance(self):
        X, W, bandwidths = self._case()
        omega = _random(3, 5, 5)
        np.testing.assert_allclose(
            kernel_distance_squared_matrix(X, W, bandwidths, jnp.tile(omega[None], (4, 1, 1))),
            kernel_distance_squared_matrix(X, W, bandwidths, omega), rtol=1e-5, atol=1e-6)

    def test_each_prototype_uses_its_own_matrix(self):
        X, W, bandwidths = self._case()
        omegas = _random(4, 4, 5, 5)
        local = kernel_distance_squared_matrix(X, W, bandwidths, omegas)
        for k in range(4):
            np.testing.assert_allclose(
                local[:, k],
                kernel_distance_squared_matrix(X, W, bandwidths, omegas[k])[:, k],
                rtol=1e-5, atol=1e-6)


class TestPushforwardWassersteinDistance:

    def _case(self, n=9, p=4, d=5):
        return _random(0, n, d), _random(1, p, d), 0.5 * _random(2, p, d)

    def test_identity_gives_the_plain_distance(self):
        X, means, log_variances = self._case()
        np.testing.assert_allclose(
            wasserstein2_pushforward_distance_matrix(X, means, log_variances, jnp.eye(5)),
            wasserstein2_distance_matrix(X, means, log_variances), rtol=1e-5)

    def test_is_the_distance_between_the_mapped_measures(self):
        """Dirac at O^T x against N(O^T mu, O^T Sigma O), with dense covariances."""
        X, means, log_variances = self._case()
        omega = _random(3, 5, 3)
        distances = wasserstein2_pushforward_distance_matrix(X, means, log_variances, omega)
        for k in range(means.shape[0]):
            covariance = omega.T @ jnp.diag(jnp.exp(log_variances[k])) @ omega
            expected = (jnp.sum(((X - means[k]) @ omega) ** 2, axis=1)
                        + jnp.trace(covariance))
            np.testing.assert_allclose(distances[:, k], expected, rtol=1e-4)

    def test_is_the_expected_adaptive_distance_to_a_draw_of_the_prototype(self):
        X, means, log_variances = self._case(n=3, p=2, d=4)
        omega = _random(3, 4, 4)
        noise = _random(5, 200_000, 4)
        distances = wasserstein2_pushforward_distance_matrix(X, means, log_variances, omega)
        for k in range(2):
            draws = means[k] + jnp.exp(0.5 * log_variances[k]) * noise
            for i in range(3):
                expected = jnp.mean(jnp.sum(((X[i] - draws) @ omega) ** 2, axis=1))
                assert float(distances[i, k]) == pytest.approx(float(expected), rel=2e-2)

    def test_variance_of_a_feature_the_metric_ignores_is_free(self):
        X, means, log_variances = self._case()
        omega = jnp.eye(5).at[2, 2].set(0.0)
        changed = log_variances.at[:, 2].add(3.0)
        np.testing.assert_allclose(
            wasserstein2_pushforward_distance_matrix(X, means, changed, omega),
            wasserstein2_pushforward_distance_matrix(X, means, log_variances, omega),
            rtol=1e-6)

    def test_equal_local_matrices_give_the_global_distance(self):
        X, means, log_variances = self._case()
        omega = _random(3, 5, 5)
        np.testing.assert_allclose(
            wasserstein2_pushforward_distance_matrix(
                X, means, log_variances, jnp.tile(omega[None], (4, 1, 1))),
            wasserstein2_pushforward_distance_matrix(X, means, log_variances, omega),
            rtol=1e-5)


# ---------------------------------------------------------------------------
# Kernel and Wasserstein models
# ---------------------------------------------------------------------------

class TestEuclideanMatrixModels:

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_registered(self, cls, plain, local):
        assert _MODEL_REGISTRY[cls.__name__] is cls
        assert issubclass(cls, SoftMetricMixin)
        assert issubclass(cls, SoftAssignmentMixin)
        assert issubclass(cls, plain)
        assert cls._local_metric is local

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    @pytest.mark.parametrize('use_scan', [True, False])
    def test_learns_iris(self, iris, cls, plain, local, use_scan):
        X, y = iris
        model = cls(n_prototypes_per_class=2, max_iter=100, use_scan=use_scan)
        model.fit(X, y)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]
        assert float(jnp.mean(model.predict(X) == y)) >= 0.95

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_starts_from_the_model_without_a_matrix(self, iris, cls, plain, local):
        X, y = iris
        with_matrix = cls(n_prototypes_per_class=2, max_iter=3)
        with_matrix.fit(X, y)
        without = plain(n_prototypes_per_class=2, max_iter=3)
        without.fit(X, y)
        assert float(with_matrix.loss_history_[0]) == pytest.approx(
            float(without.loss_history_[0]), rel=1e-5)

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_frozen_matrix_gives_the_model_without_a_matrix(self, iris, cls, plain, local):
        X, y = iris
        key = 'omegas' if local else 'omega'
        frozen = cls(n_prototypes_per_class=2, max_iter=25, freeze_params=[key])
        frozen.fit(X, y)
        without = plain(n_prototypes_per_class=2, max_iter=25)
        without.fit(X, y)
        identity = jnp.eye(4)
        np.testing.assert_allclose(
            frozen.omega_matrix, jnp.broadcast_to(identity, frozen.omega_matrix.shape))
        np.testing.assert_allclose(frozen.prototypes_, without.prototypes_, atol=1e-4)
        np.testing.assert_allclose(
            frozen.predict_proba(X), without.predict_proba(X), atol=1e-4)

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_matrix_is_learned(self, iris, cls, plain, local):
        X, y = iris
        model = cls(n_prototypes_per_class=2, max_iter=60)
        model.fit(X, y)
        omega = model.omega_matrix
        assert omega.shape == ((6, 4, 4) if local else (4, 4))
        assert float(jnp.max(jnp.abs(omega - jnp.eye(4)))) > 1e-2
        lam = model.lambda_matrix
        assert lam.shape == ((6, 4, 4) if local else (4, 4))
        np.testing.assert_allclose(lam, jnp.swapaxes(lam, -1, -2), atol=1e-6)
        assert float(jnp.min(jnp.linalg.eigvalsh(lam))) > -1e-5
        if local:
            # The prototypes do not share one matrix.
            assert float(jnp.max(jnp.abs(omega[0] - omega[-1]))) > 1e-3

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_latent_dim_limits_the_rank(self, iris, cls, plain, local):
        X, y = iris
        model = cls(latent_dim=2, n_prototypes_per_class=2, max_iter=60)
        model.fit(X, y)
        assert model.omega_matrix.shape[-2:] == (4, 2)
        eigenvalues = jnp.linalg.eigvalsh(model.lambda_matrix)
        assert float(jnp.max(jnp.abs(eigenvalues[..., :2]))) < 1e-4
        assert float(jnp.mean(model.predict(X) == y)) >= 0.8

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    @pytest.mark.parametrize('latent_dim', [0, -1, 2.5, 'full'])
    def test_invalid_latent_dim(self, cls, plain, local, latent_dim):
        with pytest.raises(ValueError, match='latent_dim'):
            cls(latent_dim=latent_dim)

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_matrix_is_not_available_before_fit(self, cls, plain, local):
        with pytest.raises(ValueError, match='not fitted'):
            cls().omega_matrix
        with pytest.raises(ValueError, match='not fitted'):
            cls().lambda_matrix

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_probabilities_and_rejection(self, iris, cls, plain, local):
        X, y = iris
        model = cls(n_prototypes_per_class=2, max_iter=40)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_array_equal(model.predict(X), jnp.argmax(proba, axis=1))
        np.testing.assert_array_equal(model.predict_with_rejection(X, 0.0), model.predict(X))
        assert bool(jnp.all(model.predict_with_rejection(X, 1.1) == -1))

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_sigma_is_learned_or_kept(self, iris, cls, plain, local):
        X, y = iris
        fixed = cls(max_iter=30, sigma=0.8, adapt_sigma=False)
        fixed.fit(X, y)
        assert float(fixed.sigma_) == pytest.approx(0.8, rel=1e-6)
        adapted = cls(max_iter=30, sigma=0.8)
        adapted.fit(X, y)
        assert float(adapted.sigma_) != pytest.approx(0.8, rel=1e-3)

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_save_load(self, iris, cls, plain, local):
        X, y = iris
        model = cls(latent_dim=3, n_prototypes_per_class=2, max_iter=30,
                    rejection_confidence=0.6)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = cls.load(path)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-5)
        np.testing.assert_allclose(loaded.omega_matrix, model.omega_matrix)
        assert float(loaded.sigma_) == pytest.approx(float(model.sigma_))
        assert loaded.latent_dim == 3
        assert loaded.rejection_confidence == 0.6

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_resume(self, iris, cls, plain, local):
        X, y = iris
        model = cls(max_iter=20)
        model.fit(X, y)
        first = float(model.loss_history_[-1])
        omega = model.omega_matrix
        model.fit(X, y, resume=True)
        assert float(model.loss_history_[-1]) <= first + 1e-4
        assert model.omega_matrix.shape == omega.shape

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_onnx_export_is_refused(self, iris, cls, plain, local):
        pytest.importorskip('onnx')
        X, y = iris
        model = cls(max_iter=5)
        model.fit(X, y)
        with pytest.raises(NotImplementedError, match=cls.__name__):
            model.export_onnx()


class TestMatrixLearningRate:

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_matrix_update_is_scaled_and_the_others_are_not(self, cls, plain, local):
        key = 'omegas' if local else 'omega'
        params = {'prototypes': jnp.ones((2, 3)), key: jnp.ones((3, 3)),
                  'log_sigma': jnp.asarray(0.0)}
        grads = {k: jnp.ones_like(v) for k, v in params.items()}
        updates = {}
        for ratio in (1.0, 0.25):
            optimizer = cls(matrix_lr_ratio=ratio, optimizer='sgd', lr=0.1)._optimizer
            updates[ratio], _ = optimizer.update(grads, optimizer.init(params), params)
        np.testing.assert_allclose(updates[0.25][key], 0.25 * updates[1.0][key])
        np.testing.assert_allclose(updates[1.0][key], -0.1)
        for name in ('prototypes', 'log_sigma'):
            np.testing.assert_allclose(updates[0.25][name], updates[1.0][name])

    @pytest.mark.parametrize('cls, plain, local', EUCLIDEAN)
    def test_default_and_persistence(self, iris, cls, plain, local):
        X, y = iris
        assert cls().matrix_lr_ratio == 0.3
        model = cls(matrix_lr_ratio=0.5, max_iter=5)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            assert cls.load(path).matrix_lr_ratio == 0.5

    @pytest.mark.parametrize('ratio', [0, -0.1, 'slow'])
    def test_invalid_ratio(self, ratio):
        with pytest.raises(ValueError, match='matrix_lr_ratio'):
            DKMRSLVQ(matrix_lr_ratio=ratio)
        with pytest.raises(ValueError, match='matrix_lr_ratio'):
            RiemannianMRSLVQ(manifold=SPD(3), matrix_lr_ratio=ratio)

    def test_slower_matrix_moves_less(self, iris):
        X, y = iris
        moved = {}
        for ratio in (1.0, 0.1):
            model = WMRSLVQ(matrix_lr_ratio=ratio, max_iter=20)
            model.fit(X, y)
            moved[ratio] = float(jnp.max(jnp.abs(model.omega_matrix - jnp.eye(4))))
        assert moved[0.1] < 0.5 * moved[1.0]


class TestWhatTheMatrixLearns:

    @pytest.mark.parametrize('cls', [DKMRSLVQ, WMRSLVQ])
    def test_matrix_finds_the_discriminative_direction(self, cls):
        """Two classes that differ along the direction of small variance."""
        rng = np.random.default_rng(0)
        n = 120
        points = rng.normal(size=(2 * n, 2)) * np.array([4.0, 0.5])
        points[n:, 1] += 1.6
        rotation = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
        X = jnp.asarray(points @ rotation.T, dtype=jnp.float32)
        y = jnp.asarray([0] * n + [1] * n)

        model = cls(max_iter=200, lr=0.02)
        model.fit(X, y)
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(model.lambda_matrix))
        assert abs(eigenvectors[:, -1] @ rotation[:, 1]) >= 0.98
        assert eigenvalues[-1] >= 5.0 * eigenvalues[0]
        assert float(jnp.mean(model.predict(X) == y)) >= 0.9

    @pytest.fixture(scope='class')
    def concentric(self):
        """A narrow class inside a wide one, both centred at the origin."""
        rng = np.random.default_rng(1)
        X = np.vstack([0.5 * rng.normal(size=(150, 3)), 3.0 * rng.normal(size=(150, 3))])
        return jnp.asarray(X, dtype=jnp.float32), jnp.asarray([0] * 150 + [1] * 150)

    @pytest.mark.parametrize('cls', [DKLMRSLVQ, WLMRSLVQ])
    def test_local_matrices_separate_concentric_classes(self, concentric, cls):
        X, y = concentric
        model = cls(max_iter=200, lr=0.02)
        model.fit(X, y)
        assert float(jnp.mean(model.predict(X) == y)) >= 0.93

    @pytest.mark.parametrize('cls', [WRSLVQ, WMRSLVQ])
    def test_one_shared_metric_cannot_separate_concentric_classes(self, concentric, cls):
        """With one prototype per class and one metric the boundary is a hyperplane."""
        X, y = concentric
        model = cls(max_iter=200, lr=0.02)
        model.fit(X, y)
        assert float(jnp.mean(model.predict(X) == y)) <= 0.75


# ---------------------------------------------------------------------------
# Riemannian models
# ---------------------------------------------------------------------------

class TestTangentCoordinates:

    @pytest.mark.parametrize('manifold', MANIFOLDS)
    def test_norm_is_the_geodesic_distance(self, manifold):
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        w, x = manifold.random_point(k1), manifold.random_point(k2)
        coordinates = _tangent_coordinates(manifold, w, x, 'exact')
        assert coordinates.shape == manifold.point_shape
        assert float(jnp.sum(coordinates ** 2)) == pytest.approx(
            float(manifold.distance_squared(w, x)), rel=2e-3, abs=1e-5)

    def test_so_coordinates_are_left_invariant(self):
        so = SO(3)
        keys = jax.random.split(jax.random.PRNGKey(4), 3)
        w, x, g = (so.random_point(k) for k in keys)
        np.testing.assert_allclose(
            _tangent_coordinates(so, g @ w, g @ x), _tangent_coordinates(so, w, x),
            atol=1e-4)

    def test_so_coordinates_are_skew(self):
        so = SO(3)
        k1, k2 = jax.random.split(jax.random.PRNGKey(5))
        coordinates = _tangent_coordinates(so, so.random_point(k1), so.random_point(k2))
        np.testing.assert_allclose(coordinates, -coordinates.T, atol=1e-5)

    def test_spd_coordinates_ignore_a_rescaling(self):
        spd = SPD(3)
        k1, k2 = jax.random.split(jax.random.PRNGKey(6))
        w, x = spd.random_point(k1), spd.random_point(k2)
        np.testing.assert_allclose(
            _tangent_coordinates(spd, 30.0 * w, 30.0 * x), _tangent_coordinates(spd, w, x),
            atol=1e-4)

    def test_poincare_coordinates_at_the_origin(self):
        """At the origin u = 2 artanh(|x|) x / |x|."""
        ball = HyperbolicPoincare(3)
        x = jnp.array([0.3, -0.2, 0.4])
        norm = jnp.linalg.norm(x)
        np.testing.assert_allclose(
            _tangent_coordinates(ball, jnp.zeros(3), x),
            2.0 * jnp.arctanh(norm) * x / norm, rtol=1e-4)


class TestRiemannianMatrixModels:

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_registered(self, cls, local):
        assert _MODEL_REGISTRY[cls.__name__] is cls
        assert issubclass(cls, SoftMetricMixin)
        assert issubclass(cls, RiemannianRSLVQ)
        assert cls._local_metric is local

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    @pytest.mark.parametrize('manifold', MANIFOLDS)
    def test_trains_on_the_manifold(self, cls, local, manifold):
        X, y = _manifold_data(manifold)
        model = cls(manifold=manifold, n_prototypes_per_class=2, max_iter=30, lr=0.05)
        model.fit(X, y)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]
        protos = model.prototypes_.reshape(-1, *manifold.point_shape)
        assert all(bool(manifold.belongs(p)) for p in protos)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        d_flat = X.shape[1]
        assert model.omega_matrix.shape == ((4, d_flat, d_flat) if local else (d_flat, d_flat))
        assert float(jnp.max(jnp.abs(model.omega_matrix - jnp.eye(d_flat)))) > 1e-3

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_trains_with_scan(self, cls, local):
        spd = SPD(3)
        X, y = _manifold_data(spd)
        model = cls(manifold=spd, max_iter=15, use_scan=True)
        model.fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.loss_history_)))

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    @pytest.mark.parametrize('manifold', [
        pytest.param(SPD(3), id='SPD3'), pytest.param(HyperbolicPoincare(3), id='Poincare3'),
    ])
    def test_frozen_matrix_gives_the_model_without_a_matrix(self, cls, local, manifold):
        X, y = _manifold_data(manifold)
        key = 'omegas' if local else 'omega'
        frozen = cls(manifold=manifold, n_prototypes_per_class=2, max_iter=15,
                     freeze_params=[key])
        frozen.fit(X, y)
        without = RiemannianRSLVQ(manifold=manifold, n_prototypes_per_class=2, max_iter=15)
        without.fit(X, y)
        np.testing.assert_allclose(frozen.prototypes_, without.prototypes_, atol=2e-3)
        np.testing.assert_allclose(
            frozen.predict_proba(X), without.predict_proba(X), atol=5e-3)

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_so_scores_are_left_invariant(self, cls, local):
        """One matrix means the same at every prototype: rotate everything."""
        so = SO(3)
        X, y = _manifold_data(so)
        model = cls(manifold=so, n_prototypes_per_class=2, max_iter=15)
        model.fit(X, y)
        params = model._fitted_params()
        g = so.random_point(jax.random.PRNGKey(9))
        moved = {**params, 'prototypes': jnp.einsum(
            'ij,pjk->pik', g, params['prototypes'].reshape(-1, 3, 3)).reshape(-1, 9)}
        X_moved = jnp.einsum('ij,njk->nik', g, X.reshape(-1, 3, 3)).reshape(-1, 9)
        np.testing.assert_allclose(
            model._logits(moved, X_moved), model._logits(params, X), rtol=3e-3, atol=3e-3)

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_spd_training_is_scale_equivariant(self, cls, local):
        spd = SPD(3)
        X, y = _manifold_data(spd)
        fitted = []
        for scale in (1.0, 40.0):
            model = cls(manifold=spd, n_prototypes_per_class=2, max_iter=15, lr=0.05)
            model.fit(scale * X, y)
            fitted.append(model)
        np.testing.assert_allclose(
            fitted[1].prototypes_, 40.0 * fitted[0].prototypes_, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(
            fitted[1].omega_matrix, fitted[0].omega_matrix, atol=1e-2)

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_log_map_default_follows_the_manifold(self, cls, local):
        assert cls(manifold=SO(3)).log_map == 'exact'
        assert cls(manifold=Grassmannian(4, 2)).log_map == 'first_order'
        assert cls(manifold=Grassmannian(4, 2), log_map='exact').log_map == 'exact'
        with pytest.raises(ValueError, match='log_map'):
            cls(manifold=SO(3), log_map='second_order')

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_has_no_neighbourhood_parameters(self, cls, local):
        hyperparams = cls(manifold=SPD(3))._get_hyperparams()
        for name in ('gamma_init', 'gamma_final', 'gamma_decay', 'lr_ratio', 'beta', 'margin'):
            assert name not in hyperparams
        assert hyperparams['log_map'] == 'exact'
        assert hyperparams['latent_dim'] is None

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_save_load(self, cls, local):
        gr = Grassmannian(4, 2)
        X, y = _manifold_data(gr)
        model = cls(manifold=gr, latent_dim=5, log_map='exact',
                    n_prototypes_per_class=2, max_iter=15)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = cls.load(path)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
        np.testing.assert_allclose(loaded.omega_matrix, model.omega_matrix)
        assert loaded.latent_dim == 5
        assert loaded.log_map == 'exact'
        assert loaded.update_rule == 'riemannian'

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_resume(self, cls, local):
        spd = SPD(3)
        X, y = _manifold_data(spd)
        model = cls(manifold=spd, max_iter=10)
        model.fit(X, y)
        first = float(model.loss_history_[-1])
        model.fit(X, y, resume=True)
        assert float(model.loss_history_[-1]) <= first + 1e-4

    @pytest.mark.parametrize('cls, local', RIEMANNIAN)
    def test_onnx_export_is_refused(self, cls, local):
        pytest.importorskip('onnx')
        so = SO(3)
        X, y = _manifold_data(so)
        model = cls(manifold=so, max_iter=5)
        model.fit(X, y)
        with pytest.raises(NotImplementedError, match=cls.__name__):
            model.export_onnx()

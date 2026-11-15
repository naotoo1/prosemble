"""Soft (probabilistic) variants: DKRSLVQ, WRSLVQ and RiemannianRSLVQ."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from prosemble.core.losses import (
    class_log_posterior, likelihood_ratio_loss, rslvq_loss,
)
from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.models import (
    _MODEL_REGISTRY, DKRSLVQ, WRSLVQ, RiemannianRSLVQ,
)
from prosemble.models.soft_mixin import SoftAssignmentMixin


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


EUCLIDEAN = [DKRSLVQ, WRSLVQ]
MANIFOLDS = [
    pytest.param(SO(3), id='SO3'),
    pytest.param(SPD(3), id='SPD3'),
    pytest.param(Grassmannian(4, 2), id='Gr42'),
    pytest.param(HyperbolicPoincare(3), id='Poincare3'),
]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLikelihoodRatioLoss:

    def _case(self, seed=0, n=12, p=6):
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        distances = jax.random.uniform(k1, (n, p), minval=0.1, maxval=5.0)
        proto_labels = jnp.array([0, 0, 1, 1, 2, 2])
        targets = jax.random.randint(k2, (n,), 0, 3)
        return distances, targets, proto_labels

    @pytest.mark.parametrize('sigma', [0.5, 1.0, 2.0])
    def test_equals_rslvq_loss_for_distance_scores(self, sigma):
        distances, targets, proto_labels = self._case()
        logits = -distances / (2.0 * sigma ** 2)
        np.testing.assert_allclose(
            likelihood_ratio_loss(logits, targets, proto_labels),
            rslvq_loss(distances, targets, proto_labels, sigma=sigma), rtol=1e-4)

    def test_ignores_a_constant_shared_by_all_components(self):
        distances, targets, proto_labels = self._case()
        logits = -distances
        shift = jnp.arange(logits.shape[0], dtype=jnp.float32)[:, None] * 3.0
        np.testing.assert_allclose(
            likelihood_ratio_loss(logits + shift, targets, proto_labels),
            likelihood_ratio_loss(logits, targets, proto_labels), rtol=1e-5)

    def test_is_zero_when_only_the_correct_class_has_mass(self):
        proto_labels = jnp.array([0, 1])
        targets = jnp.array([0, 1])
        logits = jnp.array([[0.0, -1e4], [-1e4, 0.0]])
        loss = likelihood_ratio_loss(logits, targets, proto_labels)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_stays_finite_for_extreme_scores(self):
        proto_labels = jnp.array([0, 1])
        targets = jnp.array([0, 0])
        logits = jnp.array([[-1e5, 0.0], [0.0, -1e5]])
        loss, grad = jax.value_and_grad(likelihood_ratio_loss)(logits, targets, proto_labels)
        assert np.isfinite(float(loss))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_class_log_posterior_is_normalised_and_matches_manual(self):
        distances, _, proto_labels = self._case()
        logits = -distances
        log_post = class_log_posterior(logits, proto_labels, 3)
        np.testing.assert_allclose(jnp.exp(log_post).sum(axis=1), 1.0, rtol=1e-5)
        probs = jax.nn.softmax(logits, axis=1)
        manual = jnp.stack([probs[:, proto_labels == c].sum(axis=1) for c in range(3)], axis=1)
        np.testing.assert_allclose(jnp.exp(log_post), manual, rtol=1e-4)


# ---------------------------------------------------------------------------
# Behaviour shared by the Euclidean soft models
# ---------------------------------------------------------------------------

class TestSoftModels:

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_registered_and_soft(self, cls):
        assert _MODEL_REGISTRY[cls.__name__] is cls
        assert issubclass(cls, SoftAssignmentMixin)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    @pytest.mark.parametrize('use_scan', [True, False])
    def test_learns_iris(self, iris, cls, use_scan):
        X, y = iris
        model = cls(n_prototypes_per_class=2, max_iter=100, use_scan=use_scan)
        model.fit(X, y)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]
        assert float(jnp.mean(model.predict(X) == y)) >= 0.93

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_probabilities(self, iris, cls):
        X, y = iris
        model = cls(n_prototypes_per_class=2, max_iter=50)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_array_equal(model.predict(X), jnp.argmax(proba, axis=1))

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_sigma_is_learned_or_kept(self, iris, cls):
        X, y = iris
        fixed = cls(max_iter=40, sigma=0.8, adapt_sigma=False)
        fixed.fit(X, y)
        assert float(fixed.sigma_) == pytest.approx(0.8, rel=1e-6)

        adapted = cls(max_iter=40, sigma=0.8)
        adapted.fit(X, y)
        assert float(adapted.sigma_) > 0
        assert float(adapted.sigma_) != pytest.approx(0.8, rel=1e-3)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_auto_sigma_follows_the_scale_of_the_data(self, iris, cls):
        X, y = iris
        small = cls(max_iter=2, adapt_sigma=False)
        small.fit(X, y)
        large = cls(max_iter=2, adapt_sigma=False)
        large.fit(10.0 * X, y)
        assert float(small.sigma_) > 0
        if cls is WRSLVQ:
            # Squared distances scale by 100, so sigma scales by 10.
            assert float(large.sigma_) == pytest.approx(10.0 * float(small.sigma_), rel=1e-3)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    @pytest.mark.parametrize('sigma', [0.0, -1.0, 'median'])
    def test_invalid_sigma(self, cls, sigma):
        with pytest.raises(ValueError, match='sigma'):
            cls(sigma=sigma)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_labels_are_decoded(self, iris, cls):
        X, y = iris
        relabelled = jnp.asarray(np.array([3, 7, 11])[np.asarray(y)])
        model = cls(max_iter=40)
        model.fit(X, relabelled)
        assert set(np.unique(np.asarray(model.predict(X)))) <= {3, 7, 11}
        assert float(jnp.mean(model.predict(X) == relabelled)) >= 0.8

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_rejection(self, iris, cls):
        X, y = iris
        model = cls(max_iter=40)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict_with_rejection(X), model.predict(X))
        np.testing.assert_array_equal(model.predict_with_rejection(X, 0.0), model.predict(X))
        assert bool(jnp.all(model.predict_with_rejection(X, 1.1) == -1))

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_save_load(self, iris, cls):
        X, y = iris
        model = cls(n_prototypes_per_class=2, max_iter=30, rejection_confidence=0.6)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = cls.load(path)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-5)
        assert float(loaded.sigma_) == pytest.approx(float(model.sigma_))
        assert loaded.rejection_confidence == 0.6
        assert loaded.adapt_sigma is True

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_resume(self, iris, cls):
        X, y = iris
        model = cls(max_iter=20)
        model.fit(X, y)
        first = float(model.loss_history_[-1])
        model.fit(X, y, resume=True)
        assert float(model.loss_history_[-1]) <= first + 1e-4

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_onnx_export_is_refused(self, iris, cls):
        pytest.importorskip('onnx')
        X, y = iris
        model = cls(max_iter=5)
        model.fit(X, y)
        with pytest.raises(NotImplementedError, match=cls.__name__):
            model.export_onnx()


# ---------------------------------------------------------------------------
# Model-specific mathematics
# ---------------------------------------------------------------------------

class TestWRSLVQ:

    def test_only_the_total_variance_enters(self, iris):
        X, y = iris
        model = WRSLVQ(max_iter=20)
        model.fit(X, y)
        params = model._fitted_params()
        shuffled = {**params, 'log_variances': params['log_variances'][:, ::-1]}
        np.testing.assert_allclose(
            model._logits(shuffled, X), model._logits(params, X), rtol=1e-5, atol=1e-5)

    def test_variance_acts_as_a_log_prior(self, iris):
        X, y = iris
        model = WRSLVQ(max_iter=20)
        model.fit(X, y)
        params = model._fitted_params()
        sigma2 = float(model.sigma_) ** 2
        no_variance = {**params, 'log_variances': jnp.full_like(params['log_variances'], -30.0)}
        offset = model._logits(params, X) - model._logits(no_variance, X)
        expected = -jnp.sum(model.prototype_variances_, axis=1) / (2.0 * sigma2)
        np.testing.assert_allclose(
            offset, jnp.broadcast_to(expected, offset.shape), rtol=1e-3, atol=1e-4)


class TestDKRSLVQ:

    def test_scores_are_bounded(self, iris):
        X, y = iris
        model = DKRSLVQ(max_iter=20)
        model.fit(X, y)
        logits = model._logits(model._fitted_params(), X)
        bound = 1.0 / float(model.sigma_) ** 2
        assert float(logits.max()) <= 1e-6
        assert float(logits.min()) >= -bound - 1e-5

    def test_far_samples_get_uniform_posteriors(self, iris):
        X, y = iris
        model = DKRSLVQ(max_iter=20)
        model.fit(X, y)
        far = 1e3 * jnp.ones((1, X.shape[1]))
        np.testing.assert_allclose(model.predict_proba(far), 1.0 / 3.0, atol=1e-4)

    def test_bandwidths_stay_above_their_floor(self, iris):
        X, y = iris
        model = DKRSLVQ(max_iter=60, sigma_min=0.05)
        model.fit(X, y)
        assert float(model.kernel_bandwidths.min()) >= 0.05


class TestRiemannianRSLVQ:

    def test_registered(self):
        assert _MODEL_REGISTRY['RiemannianRSLVQ'] is RiemannianRSLVQ

    @pytest.mark.parametrize('manifold', MANIFOLDS)
    @pytest.mark.parametrize('use_scan', [False, True])
    def test_trains_on_the_manifold(self, manifold, use_scan):
        X, y = _manifold_data(manifold)
        model = RiemannianRSLVQ(manifold=manifold, n_prototypes_per_class=2,
                                max_iter=30, lr=0.05, use_scan=use_scan)
        model.fit(X, y)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]
        protos = model.prototypes_.reshape(-1, *manifold.point_shape)
        assert all(bool(manifold.belongs(p)) for p in protos)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    @pytest.mark.parametrize('update_rule', ['riemannian', 'projected'])
    def test_update_rules(self, update_rule):
        spd = SPD(3)
        X, y = _manifold_data(spd)
        model = RiemannianRSLVQ(manifold=spd, max_iter=15, update_rule=update_rule)
        model.fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.prototypes_)))

    def test_has_no_neighbourhood_parameters(self):
        model = RiemannianRSLVQ(manifold=SPD(3))
        hyperparams = model._get_hyperparams()
        for name in ('gamma_init', 'gamma_final', 'gamma_decay', 'lr_ratio', 'beta', 'margin'):
            assert name not in hyperparams
        assert model.freeze_params is None

    def test_one_prototype_per_class_predicts_the_closest_prototype(self):
        spd = SPD(3)
        X, y = _manifold_data(spd)
        model = RiemannianRSLVQ(manifold=spd, n_prototypes_per_class=1, max_iter=20)
        model.fit(X, y)
        distances = model._squared_distances(X, model.prototypes_)
        closest = model.prototype_labels_[jnp.argmin(distances, axis=1)]
        np.testing.assert_array_equal(model.predict(X), closest)

    def test_spd_training_is_scale_equivariant(self):
        """The affine-invariant distance ignores a rescaling of the data."""
        spd = SPD(3)
        X, y = _manifold_data(spd)
        fitted = []
        for scale in (1.0, 40.0):
            model = RiemannianRSLVQ(manifold=spd, n_prototypes_per_class=2,
                                    max_iter=20, lr=0.05)
            model.fit(scale * X, y)
            fitted.append((np.asarray(model.prototypes_), float(model.sigma_)))
        np.testing.assert_allclose(fitted[1][0], 40.0 * fitted[0][0], rtol=5e-3, atol=5e-3)
        assert fitted[1][1] == pytest.approx(fitted[0][1], rel=1e-3)

    def test_components_on_so_are_matrix_fisher(self):
        """Chordal scores equal tr(w^T x) / sigma^2 up to a constant."""
        so = SO(3)
        X, y = _manifold_data(so)
        model = RiemannianRSLVQ(manifold=so, max_iter=10)
        model.fit(X, y)
        logits = model._logits(model._fitted_params(), X)
        traces = jnp.einsum('nij,pij->np', X.reshape(-1, 3, 3),
                            model.prototypes_.reshape(-1, 3, 3))
        difference = logits - traces / float(model.sigma_) ** 2
        np.testing.assert_allclose(difference, difference[0, 0], atol=2e-3)

    def test_save_load(self):
        gr = Grassmannian(4, 2)
        X, y = _manifold_data(gr)
        model = RiemannianRSLVQ(manifold=gr, n_prototypes_per_class=2, max_iter=15)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = RiemannianRSLVQ.load(path)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
        assert float(loaded.sigma_) == pytest.approx(float(model.sigma_))
        assert loaded.update_rule == 'riemannian'

    def test_onnx_export_is_refused(self):
        pytest.importorskip('onnx')
        so = SO(3)
        X, y = _manifold_data(so)
        model = RiemannianRSLVQ(manifold=so, max_iter=5)
        model.fit(X, y)
        with pytest.raises(NotImplementedError, match='RiemannianRSLVQ'):
            model.export_onnx()

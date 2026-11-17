"""One-class soft variants trained by contrast with a reference.

OCDKRSLVQ, OCWRSLVQ and OCRiemannianRSLVQ.
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.models import (
    _MODEL_REGISTRY,
    OCDKRSLVQ, OCWRSLVQ, OCRiemannianRSLVQ,
    DKRSLVQ, WRSLVQ, RiemannianRSLVQ,
)
from prosemble.models.oc_soft_mixin import (
    OUTLIER, REFERENCE, TARGET, OneClassSoftMixin, RiemannianReferenceMixin,
    _log_sinhc, kmeans_init,
)

EUCLIDEAN = [OCDKRSLVQ, OCWRSLVQ]
MANIFOLDS = [
    pytest.param(SO(3), id='SO3'),
    pytest.param(SPD(3), id='SPD3'),
    pytest.param(Grassmannian(4, 2), id='Gr42'),
    pytest.param(HyperbolicPoincare(3), id='Poincare3'),
]


@pytest.fixture(scope='module')
def blobs():
    """Targets in two clusters; outliers in a third, away from both."""
    rng = np.random.default_rng(0)
    targets = np.vstack([
        rng.normal(size=(60, 4)) * 0.6 + np.array([2.0, 0.0, 0.0, 0.0]),
        rng.normal(size=(60, 4)) * 0.6 - np.array([2.0, 0.0, 0.0, 0.0]),
    ])
    outliers = rng.normal(size=(80, 4)) * 0.6 + np.array([0.0, 4.0, 0.0, 0.0])
    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    return f(targets[:90]), f(targets[90:]), f(outliers[:10]), f(outliers[10:])


def _auroc(target_scores, outlier_scores):
    """Probability that a target scores above an outlier."""
    t = np.asarray(target_scores)[:, None]
    o = np.asarray(outlier_scores)[None, :]
    return float(np.mean((t > o) + 0.5 * (t == o)))


def _tangent_gaussian(manifold, centre, scale, n, seed):
    """Points around ``centre``: Gaussian tangent coordinates, mapped by Exp."""
    noise = scale * jax.random.normal(jax.random.PRNGKey(seed), (n, *manifold.point_shape))
    points = jax.vmap(
        lambda u: manifold.exp_map(centre, manifold.frame_to_tangent(centre, u)))(noise)
    return points.reshape(n, -1)


def _manifold_one_class_data(manifold, seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    centre, far = manifold.random_point(k1), manifold.random_point(k2)
    if isinstance(manifold, HyperbolicPoincare):
        centre, far = jnp.array([0.3, 0.1, -0.2]), jnp.array([-0.5, -0.4, 0.3])
    targets = _tangent_gaussian(manifold, centre, 0.15, 50, seed + 1)
    outliers = _tangent_gaussian(manifold, far, 0.15, 40, seed + 2)
    return targets[:35], targets[35:], outliers


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TestKMeansInit:

    def test_finds_the_cluster_centres_of_each_class(self, blobs):
        train, _, labelled, _ = blobs
        X = jnp.concatenate([train, labelled])
        y = jnp.concatenate([jnp.zeros(90, dtype=jnp.int32), jnp.ones(10, dtype=jnp.int32)])
        prototypes, labels = kmeans_init(X, y, [2, 1], jax.random.PRNGKey(0))
        assert labels.tolist() == [0, 0, 1]
        first = sorted(float(v) for v in prototypes[:2, 0])
        assert first[0] == pytest.approx(-2.0, abs=0.4)
        assert first[1] == pytest.approx(2.0, abs=0.4)
        assert float(prototypes[2, 1]) == pytest.approx(4.0, abs=0.6)

    def test_too_few_samples(self, blobs):
        train = blobs[0]
        with pytest.raises(ValueError, match='fewer than'):
            kmeans_init(train[:2], jnp.zeros(2, dtype=jnp.int32), 3, jax.random.PRNGKey(0))


class TestLogSinhc:

    def test_matches_the_direct_formula_and_its_limits(self):
        t = jnp.array([0.0, 1e-4, 0.5, 3.0, 20.0])
        direct = jnp.log(jnp.sinh(t[2:]) / t[2:])
        np.testing.assert_allclose(_log_sinhc(t)[2:], direct, rtol=1e-5)
        np.testing.assert_allclose(_log_sinhc(t)[:2], 0.0, atol=1e-7)
        # log(sinh t / t) -> t - log(2t) without overflow.
        assert float(_log_sinhc(jnp.asarray(200.0))) == pytest.approx(
            200.0 - np.log(400.0), rel=1e-5)


class TestEuclideanReference:

    def test_is_a_gaussian_around_the_targets(self, blobs):
        train = blobs[0]
        model = OCWRSLVQ(noise_scale=2.0)
        model._fit_reference(train)
        np.testing.assert_allclose(model.reference_loc_, jnp.mean(train, axis=0), atol=1e-5)
        np.testing.assert_allclose(
            model.reference_scale_, 2.0 * jnp.std(train, axis=0), rtol=1e-5)
        samples = model._sample_reference(jax.random.PRNGKey(0), 20000)
        np.testing.assert_allclose(jnp.std(samples, axis=0), model.reference_scale_, rtol=0.05)

    def test_log_density_is_gaussian_up_to_a_constant(self, blobs):
        train = blobs[0]
        model = OCWRSLVQ()
        model._fit_reference(train)
        z = (train - model.reference_loc_) / model.reference_scale_
        expected = jax.scipy.stats.norm.logpdf(z).sum(axis=1) - jnp.sum(
            jnp.log(model.reference_scale_))
        difference = model._reference_log_density(train) - expected
        np.testing.assert_allclose(difference, difference[0], atol=1e-4)

    def test_constant_feature_keeps_a_finite_width(self, blobs):
        train = blobs[0].at[:, 3].set(1.5)
        model = OCWRSLVQ()
        model._fit_reference(train)
        assert float(model.reference_scale_[3]) > 0.1
        shifted = train.at[:, 3].set(2.5)
        assert bool(jnp.all(jnp.isfinite(model._reference_log_density(shifted))))
        assert float(jnp.min(model._reference_log_density(shifted))) > -1e3


# ---------------------------------------------------------------------------
# Kernel and Wasserstein models
# ---------------------------------------------------------------------------

class TestEuclideanOneClassModels:

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_registered(self, cls):
        assert _MODEL_REGISTRY[cls.__name__] is cls
        assert issubclass(cls, OneClassSoftMixin)
        assert cls._encode_labels is False

    @pytest.mark.parametrize('cls, parent', [(OCDKRSLVQ, DKRSLVQ), (OCWRSLVQ, WRSLVQ)])
    def test_uses_the_distance_of_the_multi_class_model(self, cls, parent):
        assert issubclass(cls, parent)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    @pytest.mark.parametrize('use_scan', [True, False])
    def test_learns_from_targets_alone(self, blobs, cls, use_scan):
        train, held_out, _, outliers = blobs
        model = cls(n_prototypes=2, max_iter=120, lr=0.02, use_scan=use_scan)
        model.fit(train)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]
        assert model.prototype_labels_.tolist() == [TARGET, TARGET]
        assert _auroc(model.score_samples(held_out), model.score_samples(outliers)) >= 0.97
        assert float(jnp.mean(model.predict(outliers) == model._non_target_label)) >= 0.9

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_targets_alone_have_a_finite_optimum(self, blobs, cls):
        """The cost does not improve by accepting everything."""
        train, held_out, _, _ = blobs
        model = cls(n_prototypes=2, max_iter=400, lr=0.02)
        model.fit(train)
        history = np.asarray(model.loss_history_)
        assert abs(history[-1] - history[-50]) < 0.02
        far = 30.0 * jnp.ones((5, 4))
        assert bool(jnp.all(model.score_samples(far) < jnp.min(model.score_samples(held_out))))
        assert bool(jnp.all(model.predict(far) == model._non_target_label))

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_nu_sets_the_rejected_fraction_of_training_targets(self, blobs, cls):
        train = blobs[0]
        for nu in (0.05, 0.3):
            model = cls(n_prototypes=2, max_iter=60, lr=0.02, nu=nu)
            model.fit(train)
            rejected = float(jnp.mean(model.predict(train) == model._non_target_label))
            assert rejected == pytest.approx(nu, abs=0.03)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_decision_function(self, blobs, cls):
        train, held_out, _, outliers = blobs
        model = cls(n_prototypes=2, max_iter=60, lr=0.02)
        model.fit(train)
        scores = model.decision_function(jnp.concatenate([held_out, outliers]))
        assert scores.shape == (held_out.shape[0] + outliers.shape[0],)
        assert bool(jnp.all((scores >= 0) & (scores <= 1)))
        np.testing.assert_array_equal(model.predict_proba(held_out),
                                      model.decision_function(held_out))
        expected = jnp.where(scores >= 0.5, model._target_label, model._non_target_label)
        np.testing.assert_array_equal(
            model.predict(jnp.concatenate([held_out, outliers])), expected)
        # Without labelled outliers the score orders samples by density.
        order = jnp.argsort(model.score_samples(held_out))
        assert bool(jnp.all(jnp.diff(model.decision_function(held_out)[order]) >= -1e-6))

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_reject_options(self, blobs, cls):
        train, held_out, _, outliers = blobs
        model = cls(n_prototypes=2, max_iter=60, lr=0.02)
        model.fit(train)
        X = jnp.concatenate([held_out, outliers])
        np.testing.assert_array_equal(model.predict_with_reject(X), model.predict(X))
        np.testing.assert_array_equal(model.predict_with_rejection(X), model.predict(X))
        assert bool(jnp.all(model.predict_with_reject(X, upper=1.1, lower=-0.1) == -1))
        rejecting = model.predict_with_rejection(X, confidence=0.999)
        assert set(np.unique(np.asarray(rejecting)).tolist()) <= {
            -1, model._target_label, model._non_target_label}

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_labelled_outliers_get_their_own_prototypes(self, blobs, cls):
        train, held_out, labelled, outliers = blobs
        X = jnp.concatenate([train, labelled])
        y = jnp.concatenate([jnp.full(90, 7), jnp.full(10, 3)])
        model = cls(n_prototypes=2, n_outlier_prototypes=2, max_iter=120, lr=0.02)
        model.fit(X, y)
        assert model.prototype_labels_.tolist() == [TARGET, TARGET, OUTLIER, OUTLIER]
        assert model._target_label == 7 and model._non_target_label == 3
        assert bool(jnp.all(jnp.isfinite(model.loss_history_)))
        assert set(np.unique(np.asarray(model.predict(held_out))).tolist()) <= {7, 3}
        assert _auroc(model.decision_function(held_out),
                      model.decision_function(outliers)) >= 0.97
        assert float(jnp.mean(model.predict(held_out) == 7)) >= 0.8
        assert float(jnp.mean(model.predict(outliers) == 3)) >= 0.95

    def test_outlier_prototypes_are_limited_by_the_labelled_outliers(self, blobs):
        train, _, labelled, _ = blobs
        X = jnp.concatenate([train, labelled[:2]])
        y = jnp.concatenate([jnp.zeros(90, dtype=jnp.int32), jnp.ones(2, dtype=jnp.int32)])
        model = OCWRSLVQ(n_prototypes=2, n_outlier_prototypes=5, max_iter=10, target_label=0)
        model.fit(X, y)
        assert model.prototype_labels_.tolist() == [TARGET, TARGET, OUTLIER, OUTLIER]

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_labelled_outliers_inside_the_target_region_are_rejected(self, cls):
        """A compact outlier group inside a wide target class."""
        rng = np.random.default_rng(3)
        f = lambda a: jnp.asarray(a, dtype=jnp.float32)
        targets = f(rng.normal(size=(200, 3)) * 2.0)
        inside = f(rng.normal(size=(60, 3)) * 0.25 + np.array([1.0, 1.0, 0.0]))
        alone = cls(n_prototypes=2, max_iter=200, lr=0.02)
        alone.fit(targets[:150])
        informed = cls(n_prototypes=2, n_outlier_prototypes=1, max_iter=200, lr=0.02,
                       target_label=0)
        informed.fit(
            jnp.concatenate([targets[:150], inside[:10]]),
            jnp.concatenate([jnp.zeros(150, dtype=jnp.int32), jnp.ones(10, dtype=jnp.int32)]))
        blind = _auroc(alone.decision_function(targets[150:]),
                       alone.decision_function(inside[10:]))
        seen = _auroc(informed.decision_function(targets[150:]),
                      informed.decision_function(inside[10:]))
        assert blind <= 0.6
        assert seen >= blind + 0.15
        if cls is OCDKRSLVQ:
            # Its bandwidths let the outlier component be narrower than
            # the target components; one shared width cannot.
            assert seen >= 0.9

    def test_outlier_weight(self, blobs):
        train, _, labelled, _ = blobs
        X = jnp.concatenate([train, labelled])
        y = jnp.concatenate([jnp.zeros(90, dtype=jnp.int32), jnp.ones(10, dtype=jnp.int32)])
        balanced = OCWRSLVQ(max_iter=3, target_label=0)
        balanced.fit(X, y)
        assert balanced._outlier_weight == pytest.approx(9.0)
        fixed = OCWRSLVQ(max_iter=3, target_label=0, outlier_weight=2.0)
        fixed.fit(X, y)
        assert fixed._outlier_weight == pytest.approx(2.0)

    def test_loss_matches_the_definition(self, blobs):
        """-log P(label | x) with the reference as a class of its own."""
        train, _, labelled, _ = blobs
        model = OCWRSLVQ(n_prototypes=2, n_outlier_prototypes=1, max_iter=5,
                         target_label=0, outlier_weight=3.0)
        model.fit(jnp.concatenate([train, labelled]),
                  jnp.concatenate([jnp.zeros(90, dtype=jnp.int32),
                                   jnp.ones(10, dtype=jnp.int32)]))
        params = model._fitted_params()
        reference = model._sample_reference(jax.random.PRNGKey(5), 40)
        X = jnp.concatenate([train[:20], labelled, reference])
        y = jnp.concatenate([jnp.full(20, TARGET), jnp.full(10, OUTLIER),
                             jnp.full(40, REFERENCE)])
        scores = np.asarray(model._component_scores(params, X), dtype=np.float64)
        offset = np.asarray(model._offset(X), dtype=np.float64)
        mass = np.exp(scores)
        denominator = mass.sum(axis=1) + np.exp(offset)
        target, outlier = mass[:, :2].sum(axis=1), mass[:, 2]
        numerator = np.where(np.asarray(y) == TARGET, target,
                             np.where(np.asarray(y) == OUTLIER, outlier, np.exp(offset)))
        weights = np.where(np.asarray(y) == OUTLIER, 3.0, 1.0)
        expected = np.sum(weights * -np.log(numerator / denominator)) / np.sum(weights)
        assert float(model._compute_loss(params, X, y, model.prototype_labels_)) == pytest.approx(
            expected, rel=1e-4)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_target_label(self, blobs, cls):
        train, _, labelled, _ = blobs
        X = jnp.concatenate([train, labelled])
        y = jnp.concatenate([jnp.full(90, 4), jnp.full(10, 9)])
        # Default: the most frequent label.
        model = cls(max_iter=3)
        model.fit(X, y)
        assert model._target_label == 4 and model._non_target_label == 9
        with pytest.raises(ValueError, match='target label'):
            cls(max_iter=3, target_label=5).fit(X, y)
        # Without labels every sample is a target.
        unlabelled = cls(max_iter=3)
        unlabelled.fit(train)
        assert unlabelled._target_label == 0 and unlabelled._non_target_label == 1

    @pytest.mark.parametrize('kwargs, match', [
        (dict(n_prototypes=0), 'n_prototypes'),
        (dict(n_outlier_prototypes=0), 'n_outlier_prototypes'),
        (dict(noise_ratio=0), 'noise_ratio'),
        (dict(noise_scale=-1.0), 'noise_scale'),
        (dict(nu=1.0), 'nu'),
        (dict(outlier_weight='heavy'), 'outlier_weight'),
        (dict(sigma=-1.0), 'sigma'),
    ])
    def test_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OCDKRSLVQ(**kwargs)

    def test_unknown_parameter(self):
        with pytest.raises(TypeError):
            OCWRSLVQ(gamma_init=1.0)

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_save_load(self, blobs, cls):
        train, held_out, labelled, outliers = blobs
        X = jnp.concatenate([train, labelled])
        y = jnp.concatenate([jnp.full(90, 7), jnp.full(10, 3)])
        model = cls(n_prototypes=2, n_outlier_prototypes=2, max_iter=30, lr=0.02,
                    noise_ratio=3, noise_scale=2.0, nu=0.1)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = cls.load(path)
        test = jnp.concatenate([held_out, outliers])
        np.testing.assert_allclose(
            loaded.decision_function(test), model.decision_function(test), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(loaded.predict(test), model.predict(test))
        assert loaded._target_label == 7 and loaded._non_target_label == 3
        assert (loaded.n_prototypes, loaded.n_outlier_prototypes) == (2, 2)
        assert (loaded.noise_ratio, loaded.noise_scale, loaded.nu) == (3, 2.0, 0.1)
        assert 'n_prototypes_per_class' not in loaded._get_hyperparams()

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_resume(self, blobs, cls):
        train = blobs[0]
        model = cls(n_prototypes=2, max_iter=30, lr=0.02)
        model.fit(train)
        first = float(model.loss_history_[-1])
        reference = (model.reference_loc_, model.reference_scale_)
        model.fit(train, resume=True)
        assert float(model.loss_history_[-1]) <= first + 1e-3
        np.testing.assert_array_equal(model.reference_loc_, reference[0])
        np.testing.assert_array_equal(model.reference_scale_, reference[1])

    @pytest.mark.parametrize('cls', EUCLIDEAN)
    def test_onnx_export_is_refused(self, blobs, cls):
        pytest.importorskip('onnx')
        model = cls(max_iter=5)
        model.fit(blobs[0])
        with pytest.raises(NotImplementedError, match=cls.__name__):
            model.export_onnx()


# ---------------------------------------------------------------------------
# Riemannian models
# ---------------------------------------------------------------------------

class TestRiemannianReference:

    @pytest.mark.parametrize('manifold', MANIFOLDS)
    def test_samples_are_points_of_the_manifold(self, manifold):
        targets = _manifold_one_class_data(manifold)[0]
        model = OCRiemannianRSLVQ(manifold=manifold)
        model._fit_reference(targets)
        samples = model._sample_reference(jax.random.PRNGKey(0), 30)
        assert samples.shape == (30, targets.shape[1])
        points = samples.reshape(30, *manifold.point_shape)
        assert all(bool(manifold.belongs(p)) for p in points)
        assert bool(jnp.all(jnp.isfinite(model._reference_log_density(samples))))

    @pytest.mark.parametrize('manifold', [
        pytest.param(SO(3), id='SO3'), pytest.param(Grassmannian(4, 2), id='Gr42')])
    def test_compact_manifolds_use_the_uniform_distribution(self, manifold):
        targets = _manifold_one_class_data(manifold)[0]
        model = OCRiemannianRSLVQ(manifold=manifold)
        model._fit_reference(targets)
        np.testing.assert_array_equal(model._reference_log_density(targets), 0.0)

    @pytest.mark.parametrize('manifold', [
        pytest.param(SPD(3), id='SPD3'), pytest.param(HyperbolicPoincare(3), id='Poincare3')])
    def test_tangent_gaussian_is_centred_and_scaled_on_the_targets(self, manifold):
        targets = _manifold_one_class_data(manifold)[0]
        model = OCRiemannianRSLVQ(manifold=manifold, noise_scale=2.0)
        model._fit_reference(targets)
        base = model.reference_loc_.reshape(manifold.point_shape)
        assert bool(manifold.belongs(base))
        # The centre is close to the targets, at the scale of their spread.
        to_targets = jax.vmap(lambda x: manifold.distance_squared(base, x))(
            targets.reshape(-1, *manifold.point_shape))
        between = jax.vmap(lambda x: manifold.distance_squared(
            targets[0].reshape(manifold.point_shape), x))(
            targets.reshape(-1, *manifold.point_shape))
        assert float(jnp.mean(to_targets)) <= float(jnp.mean(between))
        # Tangent coordinates of the samples have the fitted scale.
        samples = model._sample_reference(jax.random.PRNGKey(1), 4000)
        coords = model._reference_coordinates(samples).reshape(4000, -1)
        dim = 6 if isinstance(manifold, SPD) else 3
        measured = jnp.sqrt(jnp.mean(jnp.sum(coords ** 2, axis=1)) / dim)
        assert float(measured) == pytest.approx(float(model.reference_scale_), rel=0.05)

    def test_spd_volume_distortion_matches_the_jacobian(self):
        """Density = Gaussian of the coordinates / volume distortion of Exp."""
        spd = SPD(3)
        model = OCRiemannianRSLVQ(manifold=spd)
        model.reference_loc_ = (2.0 * jnp.eye(3) + 0.3 * jnp.ones((3, 3))).reshape(-1)
        model.reference_scale_ = jnp.asarray(0.7)
        base = model.reference_loc_.reshape(3, 3)
        basis = []
        for i in range(3):
            basis.append(jnp.zeros((3, 3)).at[i, i].set(1.0))
        for i in range(3):
            for j in range(i + 1, 3):
                basis.append(jnp.zeros((3, 3)).at[i, j].set(2 ** -0.5).at[j, i].set(2 ** -0.5))
        basis = jnp.stack(basis)

        def point(c):
            u = jnp.einsum('k,kij->ij', c, basis)
            return spd.exp_map(base, spd.frame_to_tangent(base, u))

        for seed in range(3):
            c = 0.6 * jax.random.normal(jax.random.PRNGKey(seed), (6,))
            x = point(c)
            jac = jax.jacobian(point)(c).reshape(9, 6)
            x_inv = jnp.linalg.inv(x)
            metric = jnp.kron(x_inv, x_inv)
            distortion = 0.5 * jnp.linalg.slogdet(jac.T @ metric @ jac)[1]
            expected = -0.5 * jnp.sum(c ** 2) / 0.7 ** 2 - distortion
            assert float(model._reference_log_density(x.reshape(1, -1))[0]) == pytest.approx(
                float(expected), abs=2e-3)

    def test_poincare_volume_distortion_matches_the_jacobian(self):
        ball = HyperbolicPoincare(3)
        model = OCRiemannianRSLVQ(manifold=ball)
        model.reference_loc_ = jnp.array([0.3, -0.2, 0.1])
        model.reference_scale_ = jnp.asarray(0.8)
        base = model.reference_loc_

        def point(c):
            return ball.exp_map(base, ball.frame_to_tangent(base, c))

        for seed in range(3):
            c = 0.7 * jax.random.normal(jax.random.PRNGKey(seed), (3,))
            x = point(c)
            jac = jax.jacobian(point)(c)
            conformal = 2.0 / (1.0 - jnp.sum(x ** 2))
            distortion = 0.5 * jnp.linalg.slogdet(jac.T @ jac * conformal ** 2)[1]
            expected = -0.5 * jnp.sum(c ** 2) / 0.8 ** 2 - distortion
            assert float(model._reference_log_density(x.reshape(1, -1))[0]) == pytest.approx(
                float(expected), abs=2e-3)


class TestRiemannianOneClassModels:

    def test_registered(self):
        cls = OCRiemannianRSLVQ
        assert _MODEL_REGISTRY[cls.__name__] is cls
        assert issubclass(cls, RiemannianReferenceMixin)
        assert issubclass(cls, OneClassSoftMixin)
        assert issubclass(cls, RiemannianRSLVQ)

    @pytest.mark.parametrize('manifold', MANIFOLDS)
    def test_learns_from_targets_alone(self, manifold):
        train, held_out, outliers = _manifold_one_class_data(manifold)
        model = OCRiemannianRSLVQ(manifold=manifold, n_prototypes=2, max_iter=40, lr=0.03,
                                  noise_ratio=2)
        model.fit(train)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history))
        assert history[-1] < history[0]
        protos = model.prototypes_.reshape(-1, *manifold.point_shape)
        assert all(bool(manifold.belongs(p)) for p in protos)
        assert _auroc(model.score_samples(held_out), model.score_samples(outliers)) >= 0.95
        scores = model.decision_function(outliers)
        assert bool(jnp.all((scores >= 0) & (scores <= 1)))

    def test_labelled_outliers(self):
        spd = SPD(3)
        train, held_out, outliers = _manifold_one_class_data(spd)
        X = jnp.concatenate([train, outliers[:8]])
        y = jnp.concatenate([jnp.zeros(35, dtype=jnp.int32), jnp.ones(8, dtype=jnp.int32)])
        model = OCRiemannianRSLVQ(manifold=spd, n_prototypes=2, n_outlier_prototypes=1,
                                  max_iter=25, lr=0.03, target_label=0, noise_ratio=2)
        model.fit(X, y)
        assert model.prototype_labels_.tolist() == [TARGET, TARGET, OUTLIER]
        assert float(jnp.mean(model.predict(outliers[8:]) == 1)) >= 0.95
        assert float(jnp.mean(model.predict(held_out) == 0)) >= 0.7

    def test_trains_with_scan(self):
        ball = HyperbolicPoincare(3)
        train = _manifold_one_class_data(ball)[0]
        model = OCRiemannianRSLVQ(manifold=ball, n_prototypes=2, max_iter=20, use_scan=True)
        model.fit(train)
        assert bool(jnp.all(jnp.isfinite(model.loss_history_)))

    @pytest.mark.parametrize('manifold', [
        pytest.param(SPD(3), id='SPD3'), pytest.param(Grassmannian(4, 2), id='Gr42')])
    def test_save_load(self, manifold):
        train, held_out, outliers = _manifold_one_class_data(manifold)
        model = OCRiemannianRSLVQ(manifold=manifold, n_prototypes=2, max_iter=15, nu=0.1)
        model.fit(train)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = OCRiemannianRSLVQ.load(path)
        test = jnp.concatenate([held_out, outliers])
        np.testing.assert_allclose(
            loaded.decision_function(test), model.decision_function(test), rtol=1e-4, atol=1e-5)
        assert loaded.nu == 0.1 and loaded.n_prototypes == 2
        assert type(loaded.manifold) is type(manifold)

    def test_onnx_export_is_refused(self):
        pytest.importorskip('onnx')
        so = SO(3)
        model = OCRiemannianRSLVQ(manifold=so, max_iter=5)
        model.fit(_manifold_one_class_data(so)[0])
        with pytest.raises(NotImplementedError, match='OCRiemannianRSLVQ'):
            model.export_onnx()


class TestStepScale:
    """Prototype steps are relative to the spread of the data."""

    @staticmethod
    def _spectra(n, scale, seed, shift=0.0):
        """Smooth curves in 300 dimensions whose spread is ``scale``."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1, 300)
        out = [np.sin(2 * np.pi * (t + shift + 0.02 * rng.normal())) + 0.3 * rng.normal() * t
               for _ in range(n)]
        return jnp.asarray(scale * np.array(out), dtype=jnp.float32)

    def test_small_spread_in_many_dimensions_does_not_collapse(self):
        from sklearn.metrics import roc_auc_score
        train = self._spectra(20, 0.01, seed=0)
        test = jnp.concatenate([self._spectra(20, 0.01, seed=1),
                                self._spectra(20, 0.01, seed=2, shift=0.15)])
        model = OCDKRSLVQ(n_prototypes=3, max_iter=60, lr=0.05, random_seed=0).fit(train)
        scores = np.asarray(model.score_samples(test))
        # Before, the first steps moved the prototypes far beyond the targets,
        # every kernel distance saturated at 2 and every score was the same.
        assert len(np.unique(scores)) > 30
        assert roc_auc_score(np.r_[np.ones(20), np.zeros(20)], scores) > 0.9

    def test_unit_spread_is_unchanged(self):
        model = OCDKRSLVQ(n_prototypes=2, max_iter=5)
        X = jnp.asarray(np.random.default_rng(0).normal(size=(50, 4)), dtype=jnp.float32)
        assert abs(model._step_scale(X) - 1.0) < 0.2

    def test_manifold_steps_are_not_rescaled(self):
        model = OCRiemannianRSLVQ(manifold=SO(3), max_iter=5)
        assert model._step_scale(jnp.ones((4, 9))) is None

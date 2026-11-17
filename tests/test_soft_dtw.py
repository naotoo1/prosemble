"""Soft dynamic time warping and SoftDTWRSLVQ."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.soft_dtw import (
    dtw, dtw_matrix, soft_dtw, soft_dtw_divergence, soft_dtw_divergence_matrix,
    soft_dtw_matrix,
)
from prosemble.models import (
    _MODEL_REGISTRY, SoftDTWMRSLVQ, SoftDTWRSLVQ,
)
from prosemble.models.soft_mixin import SoftAssignmentMixin


def _reference(x, y, gamma, soft=True, band=None):
    """Direct dynamic programme over the full accumulated-cost matrix."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64).T).T
    y = np.atleast_2d(np.asarray(y, dtype=np.float64).T).T
    n, m = len(x), len(y)
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs((i - 1) / max(n - 1, 1) - (j - 1) / max(m - 1, 1)) > band:
                continue
            prev = np.array([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]])
            if soft:
                z = -prev / gamma
                top = np.max(z)
                best = -gamma * (top + np.log(np.sum(np.exp(z - top))))
            else:
                best = prev.min()
            R[i, j] = np.sum((x[i - 1] - y[j - 1]) ** 2) + best
    return R[n, m]


@pytest.fixture(scope='class')
def float64():
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', False)


@pytest.mark.usefixtures('float64')
class TestSoftDTW:

    @pytest.mark.parametrize('n, m', [(1, 1), (5, 5), (7, 4), (3, 11)])
    @pytest.mark.parametrize('gamma', [0.01, 0.5, 3.0])
    def test_matches_the_direct_recursion(self, n, m, gamma):
        rng = np.random.default_rng(n * 13 + m)
        x, y = rng.normal(size=n), rng.normal(size=m)
        assert float(soft_dtw(jnp.asarray(x), jnp.asarray(y), gamma)) == pytest.approx(
            _reference(x, y, gamma), rel=1e-9, abs=1e-9)

    def test_multichannel(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=(6, 3)), rng.normal(size=(8, 3))
        assert float(soft_dtw(jnp.asarray(x), jnp.asarray(y), 0.3)) == pytest.approx(
            _reference(x, y, 0.3), rel=1e-9)

    def test_dtw_and_its_limit(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=9), rng.normal(size=12)
        hard = float(dtw(jnp.asarray(x), jnp.asarray(y)))
        assert hard == pytest.approx(_reference(x, y, 1.0, soft=False), rel=1e-9)
        assert float(soft_dtw(jnp.asarray(x), jnp.asarray(y), 1e-3)) == pytest.approx(hard, abs=0.05)
        # Soft-DTW is a soft minimum, so it lies below the hard minimum.
        assert float(soft_dtw(jnp.asarray(x), jnp.asarray(y), 1.0)) < hard

    def test_dtw_ignores_a_time_shift_that_euclidean_does_not(self):
        t = np.linspace(0, 1, 60)
        x = np.exp(-((t - 0.4) / 0.05) ** 2)
        y = np.exp(-((t - 0.6) / 0.05) ** 2)
        assert float(dtw(jnp.asarray(x), jnp.asarray(y))) < 0.1 * np.sum((x - y) ** 2)

    def test_band(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=10), rng.normal(size=10)
        for band in (0.1, 0.3):
            assert float(dtw(jnp.asarray(x), jnp.asarray(y), band=band)) == pytest.approx(
                _reference(x, y, 1.0, soft=False, band=band), rel=1e-9)
        assert float(dtw(jnp.asarray(x), jnp.asarray(y), band=0.1)) >= float(dtw(jnp.asarray(x), jnp.asarray(y)))

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        x, w = jnp.asarray(rng.normal(size=12)), jnp.asarray(rng.normal(size=10))
        grad = jax.grad(lambda v: soft_dtw(x, v, 0.5))(w)
        assert bool(jnp.all(jnp.isfinite(grad)))
        for i in (0, 4, 9):
            e = jnp.zeros(10).at[i].set(1e-6)
            fd = (soft_dtw(x, w + e, 0.5) - soft_dtw(x, w - e, 0.5)) / 2e-6
            assert float(grad[i]) == pytest.approx(float(fd), rel=1e-5, abs=1e-7)

    def test_divergence_is_zero_for_equal_series_and_positive_otherwise(self):
        rng = np.random.default_rng(4)
        X = jnp.asarray(rng.normal(size=(5, 15)))
        D = np.asarray(soft_dtw_divergence_matrix(X, X, 0.5))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)
        assert D[~np.eye(5, dtype=bool)].min() > 0
        np.testing.assert_allclose(D, D.T, atol=1e-9)
        assert float(soft_dtw_divergence(X[0], X[1], 0.5)) == pytest.approx(D[0, 1])

    def test_matrices(self):
        rng = np.random.default_rng(5)
        X, Y = jnp.asarray(rng.normal(size=(3, 7))), jnp.asarray(rng.normal(size=(4, 9)))
        S = np.asarray(soft_dtw_matrix(X, Y, 0.2))
        H = np.asarray(dtw_matrix(X, Y))
        assert S.shape == H.shape == (3, 4)
        assert S[2, 1] == pytest.approx(float(soft_dtw(X[2], Y[1], 0.2)))
        assert H[0, 3] == pytest.approx(float(dtw(X[0], Y[3])))


def _shapes(n_per_class=12, length=40, seed=0):
    """Three classes of bumps with random time shifts and noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    X, y = [], []
    for c, width in enumerate((0.04, 0.12, None)):
        for _ in range(n_per_class):
            centre = rng.uniform(0.3, 0.7)
            if width is None:
                series = np.where(np.abs(t - centre) < 0.1, 1.0, 0.0)       # plateau
            else:
                series = np.exp(-((t - centre) / width) ** 2)
            X.append(series + 0.05 * rng.normal(size=length))
            y.append(c)
    X = np.array(X, dtype=np.float32)
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return jnp.asarray(X), jnp.asarray(np.array(y))


class TestSoftDTWRSLVQ:

    def test_registered(self):
        assert _MODEL_REGISTRY['SoftDTWRSLVQ'] is SoftDTWRSLVQ
        assert issubclass(SoftDTWRSLVQ, SoftAssignmentMixin)

    @pytest.mark.parametrize('use_scan', [True, False])
    def test_learns_shifted_shapes(self, use_scan):
        X, y = _shapes()
        Xt, yt = _shapes(seed=1)
        model = SoftDTWRSLVQ(gamma=0.1, max_iter=40, lr=0.05, use_scan=use_scan)
        model.fit(X, y)
        history = np.asarray(model.loss_history_)
        assert np.all(np.isfinite(history)) and history[-1] < history[0]
        assert float(jnp.mean(model.predict(Xt) == yt)) >= 0.9
        proba = np.asarray(model.predict_proba(Xt))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        assert model.prototypes_.shape == (3, 40)

    def test_logits_are_the_divergence_up_to_a_per_sample_constant(self):
        X, y = _shapes(n_per_class=4)
        model = SoftDTWRSLVQ(gamma=0.2, max_iter=5).fit(X, y)
        logits = np.asarray(model._logits(model._fitted_params(), X), dtype=np.float64)
        D = np.asarray(model.divergences(X), dtype=np.float64)
        implied = -2.0 * float(model.sigma_) ** 2 * logits
        offset = implied - D
        np.testing.assert_allclose(offset - offset[:, :1], 0.0, atol=1e-2)
        assert D.min() >= -1e-4

    def test_multichannel_series(self):
        X, y = _shapes(n_per_class=6, length=20)
        two = jnp.stack([X, -X], axis=2).reshape(X.shape[0], -1)       # time-major
        model = SoftDTWRSLVQ(n_channels=2, max_iter=20, lr=0.05).fit(two, y)
        assert model.prototypes_.shape == (3, 40)
        assert float(jnp.mean(model.predict(two) == y)) >= 0.9
        with pytest.raises(ValueError, match='n_channels'):
            SoftDTWRSLVQ(n_channels=2, max_iter=1).fit(two[:, :-1], y)

    def test_mini_batches_and_band(self):
        X, y = _shapes()
        model = SoftDTWRSLVQ(band=0.3, batch_size=12, max_iter=10, lr=0.05).fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.loss_history_)))

    @pytest.mark.parametrize('kwargs, match', [
        (dict(gamma=0), 'gamma'), (dict(gamma=-1.0), 'gamma'), (dict(band=0), 'band'),
        (dict(band=1.5), 'band'), (dict(n_channels=0), 'n_channels'), (dict(sigma=-1.0), 'sigma'),
    ])
    def test_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SoftDTWRSLVQ(**kwargs)

    def test_save_load_and_calibration(self):
        X, y = _shapes(n_per_class=6)
        model = SoftDTWRSLVQ(gamma=0.5, band=0.5, max_iter=10, lr=0.05).fit(X, y + 3)
        model.calibrate(X, y + 3, cv=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = SoftDTWRSLVQ.load(path)
        np.testing.assert_array_equal(np.asarray(loaded.predict(X)), np.asarray(model.predict(X)))
        np.testing.assert_allclose(np.asarray(loaded.predict_proba(X)),
                                   np.asarray(model.predict_proba(X)), rtol=1e-5)
        assert (loaded.gamma, loaded.band, loaded.n_channels) == (0.5, 0.5, 1)
        assert set(np.unique(np.asarray(loaded.predict(X))).tolist()) <= {3, 4, 5}

    def test_onnx_export_is_refused(self):
        pytest.importorskip('onnx')
        X, y = _shapes(n_per_class=4)
        model = SoftDTWRSLVQ(max_iter=2).fit(X, y)
        with pytest.raises(NotImplementedError, match='SoftDTWRSLVQ'):
            model.export_onnx()


def _channels(n_per_class=10, length=30, n_channels=4, seed=0):
    """Three classes told apart by channel 0 only; the other channels are noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    X, y = [], []
    for c, width in enumerate((0.04, 0.12, None)):
        for _ in range(n_per_class):
            centre = rng.uniform(0.3, 0.7)
            if width is None:
                signal = np.where(np.abs(t - centre) < 0.1, 1.0, 0.0)
            else:
                signal = np.exp(-((t - centre) / width) ** 2)
            series = rng.normal(size=(length, n_channels))
            series[:, 0] = signal + 0.05 * rng.normal(size=length)
            X.append(series)
            y.append(c)
    X = np.array(X, dtype=np.float32)
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return jnp.asarray(X.reshape(len(X), -1)), jnp.asarray(np.array(y))


class TestSoftDTWMatrix:

    def test_registered(self):
        assert _MODEL_REGISTRY['SoftDTWMRSLVQ'] is SoftDTWMRSLVQ

    def test_metric_is_soft_dtw_of_the_projected_series(self):
        X, _ = _channels(n_per_class=2)
        model = SoftDTWMRSLVQ(gamma=0.5, n_channels=4)
        series = X.reshape(len(X), -1, 4)
        W = X[:2]
        # Identity: the model without a metric.
        params = {'prototypes': W, 'omega': jnp.eye(4)}
        np.testing.assert_allclose(
            np.asarray(model._divergence_matrix(params, X)),
            np.asarray(soft_dtw_divergence_matrix(series, series[:2], 0.5)), rtol=1e-5)
        # All weight on channel 0: the univariate divergence of that channel,
        # with the map normalised to trace 4.
        omega = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        params = {'prototypes': W, 'omega': omega}
        expected = soft_dtw_divergence_matrix(2.0 * series[..., :1], 2.0 * series[:2, :, :1], 0.5)
        np.testing.assert_allclose(np.asarray(model._divergence_matrix(params, X)),
                                   np.asarray(expected), rtol=1e-5, atol=1e-4)

    def test_learns_the_informative_channel(self):
        X, y = _channels()
        X_test, y_test = _channels(n_per_class=20, seed=1)
        kw = dict(gamma=1.0, n_channels=4, max_iter=120, lr=0.05, random_seed=0)
        plain = SoftDTWRSLVQ(**kw).fit(X, y)
        metric = SoftDTWMRSLVQ(**kw).fit(X, y)
        acc = lambda m: float(jnp.mean(m.predict(X_test) == y_test))
        assert acc(metric) > acc(plain)
        relevances = np.asarray(metric.channel_relevances)
        assert relevances.shape == (4,)
        assert np.argmax(relevances) == 0 and relevances[0] > 0.5
        np.testing.assert_allclose(relevances.sum(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(float(jnp.trace(metric.lambda_matrix)), 4.0, rtol=1e-5)

    def test_invalid_latent_dim(self):
        with pytest.raises(ValueError, match='latent_dim'):
            SoftDTWMRSLVQ(latent_dim=0)

    def test_save_load_and_calibration(self):
        cls = SoftDTWMRSLVQ
        X, y = _channels(n_per_class=5)
        model = cls(gamma=1.0, n_channels=4, latent_dim=3, max_iter=10, lr=0.05).fit(X, y)
        model.calibrate(X, y, cv=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = cls.load(path)
        np.testing.assert_allclose(np.asarray(loaded.predict_proba(X)),
                                   np.asarray(model.predict_proba(X)), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(loaded.lambda_matrix),
                                   np.asarray(model.lambda_matrix), rtol=1e-6)
        assert (loaded.latent_dim, loaded.n_channels) == (3, 4)

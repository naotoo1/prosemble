"""Tests for OCAlignmentDensity, the one-class alignment density of time series."""
import itertools
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score

from prosemble.models import _MODEL_REGISTRY, OCAlignmentDensity
from prosemble.models.oc_alignment_density import _forward, _log_density, _log_z, kmedoids


def _bumps(n, T=30, centres=(0.5,), width=0.1, shift=0.1, noise=0.3, seed=0):
    """Series with one Gaussian bump each, its centre near one of ``centres``."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    out = []
    for i in range(n):
        c = centres[i % len(centres)] + rng.uniform(-shift, shift)
        out.append(np.exp(-((t - c) / width) ** 2) + noise * rng.normal(size=T))
    return np.array(out, dtype=np.float32)


def test_registered():
    assert _MODEL_REGISTRY['OCAlignmentDensity'] is OCAlignmentDensity


def test_forward_sums_over_all_alignments():
    T, L = 5, 4
    rng = np.random.default_rng(0)
    log_e = rng.normal(size=(T, L)).astype(np.float32)
    tau = np.array([0.2, 0.5, 0.3], np.float32)
    total = 0.0
    for steps in itertools.product(range(3), repeat=T - 1):
        a = np.concatenate([[0], np.cumsum(steps)])
        if a[-1] == L - 1:
            total += np.prod(tau[list(steps)]) * np.exp(log_e[np.arange(T), a].sum())
    np.testing.assert_allclose(float(_forward(jnp.asarray(log_e), jnp.log(jnp.asarray(tau)))),
                               np.log(total), rtol=1e-5)
    # Z sums the alignment probabilities alone.
    z = sum(np.prod(tau[list(s)]) for s in itertools.product(range(3), repeat=T - 1)
            if sum(s) == L - 1)
    np.testing.assert_allclose(float(_log_z(T, L, jnp.log(jnp.asarray(tau)))), np.log(z), rtol=1e-5)


def test_one_time_point_is_a_gaussian():
    """With T = L = 1 there is one alignment, and p is the Gaussian itself."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 1, 2)).astype(np.float32)
    W = rng.normal(size=(1, 1, 2)).astype(np.float32)
    cov = np.array([[1.5, 0.3], [0.3, 0.8]], np.float32)
    got = _log_density(jnp.asarray(X), jnp.asarray(W), jnp.asarray(cov), jnp.zeros(1),
                       jnp.log(jnp.array([0.25, 0.5, 0.25])), True)
    expected = multivariate_normal(W[0, 0], cov).logpdf(X[:, 0])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5)


def test_kmedoids_finds_one_medoid_per_cluster():
    points = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    D = np.abs(points[:, None] - points[None, :])
    medoids = kmedoids(D, 2, np.random.default_rng(0))
    assert sorted(points[medoids].tolist()) == [0.1, 5.1]


def test_detects_a_shape_the_class_does_not_have():
    normal = _bumps(16, centres=(0.25, 0.75), width=0.05, noise=0.05)
    model = OCAlignmentDensity(n_prototypes=2, max_iter=20).fit(normal)
    assert model.prototypes_.shape == (2, 30, 1)
    assert model.covariances_.shape == (2, 30, 1)
    np.testing.assert_allclose(float(jnp.sum(model.step_probabilities_)), 1.0, rtol=1e-5)
    np.testing.assert_allclose(float(jnp.sum(jnp.exp(model.log_weights_))), 1.0, rtol=1e-5)
    # A bump between the two shapes, or a wide one, is anomalous.
    test = np.concatenate([_bumps(10, centres=(0.25, 0.75), width=0.05, noise=0.05, seed=1),
                           _bumps(5, centres=(0.5,), width=0.05, noise=0.05, shift=0.03, seed=2),
                           _bumps(5, centres=(0.5,), width=0.3, noise=0.05, seed=3)])
    scores = np.asarray(model.score_samples(test))
    assert roc_auc_score(np.r_[np.ones(10), np.zeros(10)], scores) > 0.9


def test_training_improves_the_likelihood():
    model = OCAlignmentDensity(max_iter=10, tol=0.0).fit(_bumps(12, seed=5))
    assert model.n_iter_ == 10
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_shared_covariance_and_decision():
    normal = _bumps(12, seed=3)
    model = OCAlignmentDensity(covariance='shared', max_iter=10, nu=0.2).fit(normal)
    assert model.covariances_.shape == (1, 1)
    d = np.asarray(model.decision_function(normal))
    assert np.all((d >= 0) & (d <= 1))
    assert set(np.unique(np.asarray(model.predict(normal)))) <= {-1, 1}
    # The fraction nu of the training series falls below the threshold.
    np.testing.assert_allclose(np.mean(d < 0.5), 0.2, atol=0.1)


def test_multivariate_series_and_save_load():
    rng = np.random.default_rng(4)
    base = _bumps(10, T=20, seed=4)
    X3 = np.stack([base, -base + 0.1 * rng.normal(size=base.shape)], axis=-1).astype(np.float32)
    model = OCAlignmentDensity(n_channels=2, max_iter=10).fit(X3)
    flat = X3.reshape(len(X3), -1)
    np.testing.assert_allclose(np.asarray(model.score_samples(flat)),
                               np.asarray(model.score_samples(X3)), rtol=1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.npz')
        model.save(path)
        loaded = OCAlignmentDensity.load(path)
    np.testing.assert_allclose(np.asarray(loaded.score_samples(X3)),
                               np.asarray(model.score_samples(X3)), rtol=1e-6)
    assert loaded.covariance == model.covariance
    assert loaded.threshold_ == model.threshold_


@pytest.mark.parametrize('kwargs, match', [
    (dict(covariance='full'), 'covariance'),
    (dict(nu=0.0), 'nu'),
    (dict(n_prototypes=0), 'n_prototypes'),
    (dict(n_channels=0), 'n_channels'),
])
def test_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OCAlignmentDensity(**kwargs)


def test_input_errors():
    with pytest.raises(ValueError, match='two training series'):
        OCAlignmentDensity().fit(_bumps(1))
    with pytest.raises(ValueError, match='T x 2'):
        OCAlignmentDensity(n_channels=2).fit(np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError, match='not fitted'):
        OCAlignmentDensity().score_samples(_bumps(2))

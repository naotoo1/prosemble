"""The prototype prior of RiemannianRSLVQ.

On training data the classes of which the model separates, the likelihood
ratio keeps growing as the prototypes move away from the data; on SPD(n)
that drift can end with matrices too ill-conditioned for float32. The
optional Riemannian Gaussian prior at each class's Karcher mean keeps the
prototypes near their classes.
"""
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.core.manifolds import SO, SPD, HyperbolicPoincare
from prosemble.models import OCRiemannianRSLVQ, RiemannianMRSLVQ, RiemannianRSLVQ


def _covariances(n_per_class, seed, channels=4, length=100):
    """Covariance matrices of mixed sources; in class k source k is stronger."""
    mixing = np.random.default_rng(0).normal(size=(channels, channels))
    rng = np.random.default_rng(seed)
    covs, labels = [], []
    for k in range(3):
        for _ in range(n_per_class):
            log_power = rng.normal(0, 0.4, channels) + rng.normal(0, 0.8)
            log_power[k] += 1.5
            sources = rng.normal(size=(channels, length)) * np.exp(log_power / 2)[:, None]
            x = mixing @ sources + 0.1 * rng.normal(size=(channels, length))
            covs.append(x @ x.T / length)
            labels.append(k)
    return (jnp.asarray(np.array(covs, dtype=np.float32).reshape(len(covs), -1)),
            jnp.asarray(np.array(labels, dtype=np.int32)))


def _distance(manifold, A, B):
    return float(manifold.distance(A, B))


def test_karcher_mean_of_commuting_matrices_is_the_geometric_mean():
    spd = SPD(3)
    model = RiemannianRSLVQ(manifold=spd)
    diag = np.array([[1.0, 2.0, 4.0], [4.0, 2.0, 1.0], [2.0, 8.0, 0.5]], np.float32)
    points = jnp.asarray(np.stack([np.diag(d) for d in diag]))
    expected = np.diag(np.exp(np.log(diag).mean(0)))
    np.testing.assert_allclose(np.asarray(model._karcher_mean(points)), expected, rtol=1e-4)


def test_prior_is_off_by_default():
    X, y = _covariances(5, seed=1)
    model = RiemannianRSLVQ(manifold=SPD(4), max_iter=2).fit(X, y)
    assert model.prototype_prior is False and model._prior is None


def test_prior_term_is_added_to_the_cost():
    X, y = _covariances(10, seed=1)
    model = RiemannianRSLVQ(manifold=SPD(4), max_iter=2, prototype_prior=True).fit(X, y)
    params = {'prototypes': model.prototypes_, 'log_sigma': jnp.log(model.sigma_)}
    labels = model.prototype_labels_
    with_prior = float(model._compute_loss(params, X, y, labels))
    centres, variances, n = model._prior
    W = model.prototypes_.reshape(-1, 4, 4)
    C = centres.reshape(-1, 4, 4)
    term = sum(_distance(model.manifold, W[k], C[k]) ** 2 / (2 * float(variances[k]))
               for k in range(len(W))) / n
    prior, model._prior = model._prior, None
    without = float(model._compute_loss(params, X, y, labels))
    model._prior = prior
    np.testing.assert_allclose(with_prior - without, term, rtol=1e-3)


def test_prototypes_stay_near_their_classes():
    X, y = _covariances(20, seed=2)
    spd = SPD(4)
    kw = dict(manifold=spd, max_iter=150, lr=0.05, random_seed=0)
    free = RiemannianRSLVQ(prototype_prior=False, **kw).fit(X, y)
    held = RiemannianRSLVQ(prototype_prior=True, **kw).fit(X, y)
    assert np.all(np.isfinite(held.loss_history_))

    def drift(model):
        centres = model._prior[0] if model._prior is not None else held._prior[0]
        W = model.prototypes_.reshape(-1, 4, 4)
        return max(_distance(spd, W[k], centres.reshape(-1, 4, 4)[k]) for k in range(len(W)))

    spread = float(np.sqrt(np.max(held._prior[1])))
    assert drift(held) < 1.5 * spread
    assert drift(free) > drift(held)
    assert float(np.mean(held.predict(X) == y)) > 0.8


@pytest.mark.parametrize('manifold', [pytest.param(SO(3), id='SO3'),
                                      pytest.param(HyperbolicPoincare(2), id='Poincare2')])
def test_other_manifolds(manifold):
    import jax
    keys = jax.random.split(jax.random.PRNGKey(0), 30)
    X = jnp.stack([manifold.random_point(k).reshape(-1) for k in keys])
    y = jnp.asarray(np.repeat([0, 1], 15))
    model = RiemannianRSLVQ(manifold=manifold, max_iter=20, prototype_prior=True).fit(X, y)
    assert np.all(np.isfinite(model.loss_history_))
    assert np.all(np.isfinite(np.asarray(model._prior[0])))


def test_save_load_and_resume_rebuild_the_prior():
    X, y = _covariances(8, seed=3)
    model = RiemannianRSLVQ(manifold=SPD(4), max_iter=5, prototype_prior=True).fit(X, y)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.npz')
        model.save(path)
        loaded = RiemannianRSLVQ.load(path)
    assert loaded.prototype_prior is True and loaded._prior is None
    loaded.fit(X, y, resume=True)
    assert loaded._prior is not None
    assert np.all(np.isfinite(loaded.loss_history_))


def test_subclasses():
    X, y = _covariances(8, seed=4)
    m = RiemannianMRSLVQ(manifold=SPD(4), max_iter=5, prototype_prior=True).fit(X, y)
    assert m.prototype_prior is True and m._prior is not None
    assert np.all(np.isfinite(m.loss_history_))
    oc = OCRiemannianRSLVQ(manifold=SPD(4), max_iter=5, prototype_prior=True).fit(X[:8])
    assert oc.prototype_prior is False and oc._prior is None

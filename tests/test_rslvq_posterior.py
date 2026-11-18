"""predict_proba of the robust soft LVQ family is the posterior it is trained on."""

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from prosemble.core.losses import rslvq_loss
from prosemble.models import (
    LMRSLVQ, LMRSLVQ_NG, MRSLVQ, MRSLVQ_NG, RSLVQ, RSLVQ_NG, SLVQ,
)


@pytest.fixture(scope='module')
def iris():
    X, y = load_iris(return_X_y=True)
    return StandardScaler().fit_transform(X).astype(np.float32), y


def _distances(model, X):
    diff = X[:, None, :] - np.asarray(model.prototypes_)[None, :, :]
    if getattr(model, 'omegas_', None) is not None:
        projected = np.einsum('npd,pdl->npl', diff, np.asarray(model.omegas_))
    elif getattr(model, 'omega_', None) is not None:
        projected = np.einsum('npd,dl->npl', diff, np.asarray(model.omega_))
    else:
        projected = diff
    return np.sum(projected ** 2, axis=2)


def _mixture_posterior(d, labels, n_classes, sigma):
    logits = -d / (2 * sigma ** 2)
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    per_class = np.stack([w[:, labels == c].sum(axis=1) for c in range(n_classes)], axis=1)
    return per_class / per_class.sum(axis=1, keepdims=True)


MODELS = [SLVQ, RSLVQ, RSLVQ_NG, MRSLVQ, LMRSLVQ, MRSLVQ_NG, LMRSLVQ_NG]


@pytest.mark.parametrize('cls', MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize('sigma', [1.0, 3.0])
def test_predict_proba_is_mixture_posterior(cls, sigma, iris):
    X, y = iris
    model = cls(sigma=sigma, n_prototypes_per_class=2, max_iter=30, random_seed=0).fit(X, y)
    expected = _mixture_posterior(_distances(model, X), np.asarray(model.prototype_labels_),
                                  model.n_classes_, sigma)
    np.testing.assert_allclose(np.asarray(model.predict_proba(X)), expected, atol=1e-5)


@pytest.mark.parametrize('cls', [RSLVQ, MRSLVQ, LMRSLVQ], ids=lambda c: c.__name__)
def test_log_loss_of_predict_proba_is_the_training_cost(cls, iris):
    """The robust soft LVQ cost is the log-loss of the class posteriors."""
    X, y = iris
    model = cls(sigma=1.0, n_prototypes_per_class=2, max_iter=30, random_seed=0).fit(X, y)
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    log_loss = -np.mean(np.log(proba[np.arange(len(y)), y]))
    cost = float(rslvq_loss(jnp.asarray(_distances(model, X)), jnp.asarray(y),
                            model.prototype_labels_, sigma=1.0))
    assert log_loss == pytest.approx(cost, rel=1e-4)


def test_width_controls_confidence(iris):
    """A wider mixture gives less confident posteriors for the same prototypes."""
    X, y = iris
    model = RSLVQ(sigma=1.0, n_prototypes_per_class=1, max_iter=30, random_seed=0).fit(X, y)
    sharp = np.asarray(model.predict_proba(X)).max(axis=1).mean()
    model.sigma = 10.0
    flat = np.asarray(model.predict_proba(X)).max(axis=1).mean()
    assert flat < sharp
    assert flat < 0.5

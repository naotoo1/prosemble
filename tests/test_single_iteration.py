"""max_iter=1 used to crash every lax.scan training loop: the convergence
check took the argmax of the loss differences, of which there are none."""
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import GLVQ, HeskesSOM, KohonenSOM, NeuralGas, OCDKRSLVQ, RSLVQ
from prosemble.models.prototype_base import scan_iterations


def _data():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(loc=c, size=(20, 4)) for c in (0.0, 3.0, 6.0)])
    y = np.repeat(np.arange(3), 20)
    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y)


@pytest.mark.parametrize('history, max_iter, expected', [
    ([3.0], 1, 1),                  # one iteration: nothing to compare
    ([3.0, 3.0, 1.0], 3, 2),        # no change between the first two
    ([3.0, 2.0, 2.0], 3, 3),        # no change between the last two
    ([3.0, 2.0, 1.0], 3, 3),        # never converged
])
def test_scan_iterations(history, max_iter, expected):
    assert scan_iterations(jnp.asarray(history), max_iter, 1e-6) == expected


@pytest.mark.parametrize('make', [
    pytest.param(lambda: GLVQ(max_iter=1), id='GLVQ'),
    pytest.param(lambda: GLVQ(max_iter=1, batch_size=16), id='GLVQ-batches'),
    pytest.param(lambda: RSLVQ(max_iter=1), id='RSLVQ'),
])
def test_supervised_single_iteration(make):
    X, y = _data()
    model = make().fit(X, y)
    assert model.n_iter_ == 1 and len(model.loss_history_) == 1
    assert model.predict(X).shape == y.shape


def test_one_class_single_iteration():
    X, _ = _data()
    model = OCDKRSLVQ(max_iter=1).fit(X[:20])
    assert model.n_iter_ == 1
    assert bool(jnp.all(jnp.isfinite(model.score_samples(X))))


@pytest.mark.parametrize('make', [
    pytest.param(lambda: NeuralGas(n_prototypes=3, max_iter=1), id='NeuralGas'),
    pytest.param(lambda: KohonenSOM(grid_height=2, grid_width=2, max_iter=1), id='KohonenSOM'),
    pytest.param(lambda: HeskesSOM(grid_height=2, grid_width=2, max_iter=1), id='HeskesSOM'),
])
def test_unsupervised_single_iteration(make):
    X, _ = _data()
    model = make().fit(X)
    assert model.n_iter_ == 1

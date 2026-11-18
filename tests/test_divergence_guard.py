"""Training stops at a non-finite loss or update and keeps the best finite parameters.

The update is made non-finite on purpose: an extra optimizer transform
multiplies every update by NaN from a given step on. This exercises each
training loop (full batch or mini-batch, compiled scan or Python loop, with
or without callbacks) independently of any model's numerics.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prosemble.core.manifolds import SPD
from prosemble.models import GLVQ, RiemannianRSLVQ


def _blobs(n_per_class=20, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(c, 1.0, (n_per_class, 2)) for c in (-2.0, 2.0)])
    y = np.repeat([0, 1], n_per_class)
    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y)


def _break_update_at(model, step):
    """From optimizer update number ``step`` on, every update is NaN."""
    model._optimizer = optax.chain(
        model._optimizer,
        optax.scale_by_schedule(lambda count: jnp.where(count >= step, jnp.nan, 1.0)))
    return model


class _Recorder:
    def __init__(self):
        self.started = self.ended = False

    def on_fit_start(self, model, X):
        self.started = True

    def on_iteration_end(self, model, info):
        pass

    def on_fit_end(self, model, info):
        self.ended = True


LOOPS = {
    'scan': dict(),
    'python': dict(use_scan=False),
    'minibatch_scan': dict(batch_size=20),
    'minibatch_python': dict(batch_size=20, use_scan=False),
    'callbacks': dict(callbacks=[_Recorder()]),
}


@pytest.mark.parametrize('loop', LOOPS)
def test_divergence_stops_training_and_keeps_best_finite_parameters(loop):
    X, y = _blobs()
    kw = dict(max_iter=30, lr=0.05, epsilon=0.0, random_seed=1, **LOOPS[loop])
    minibatch = 'batch_size' in kw
    # Two batches per epoch: update 6 is in epoch 3; full batch: iteration 6.
    model = _break_update_at(GLVQ(**kw), 6)
    with pytest.warns(RuntimeWarning, match='diverged'):
        model.fit(X, y)
    assert np.all(np.isfinite(np.asarray(model.prototypes_)))
    assert np.all(np.isfinite(np.asarray(model.loss_history_)))
    # The loss falls steadily on these blobs, so the best finite parameters
    # are the last ones before the broken update.
    reference = GLVQ(**{**kw, 'max_iter': 3 if minibatch else 6}).fit(X, y)
    np.testing.assert_allclose(np.asarray(model.prototypes_),
                               np.asarray(reference.prototypes_), rtol=1e-5, atol=1e-6)
    assert np.mean(np.asarray(model.predict(X)) == np.asarray(y)) > 0.9


@pytest.mark.parametrize('loop', LOOPS)
def test_finite_training_is_unchanged(loop):
    X, y = _blobs()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        model = GLVQ(max_iter=20, lr=0.05, random_seed=1, **LOOPS[loop]).fit(X, y)
    assert np.all(np.isfinite(np.asarray(model.prototypes_)))


def test_callbacks_still_see_the_end_of_training():
    X, y = _blobs()
    recorder = _Recorder()
    model = _break_update_at(GLVQ(max_iter=10, epsilon=0.0, callbacks=[recorder]), 0)
    with pytest.warns(RuntimeWarning, match='diverged'):
        model.fit(X, y)
    assert recorder.started and recorder.ended


def test_riemannian_rslvq_suggests_the_prototype_prior():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 3, 3))
    X = jnp.asarray((A @ A.transpose(0, 2, 1) + 3 * np.eye(3)).reshape(30, -1), dtype=jnp.float32)
    y = jnp.asarray(np.repeat([0, 1], 15))
    model = _break_update_at(RiemannianRSLVQ(manifold=SPD(3), max_iter=10, epsilon=0.0), 3)
    with pytest.warns(RuntimeWarning, match='prototype_prior=True'):
        model.fit(X, y)
    assert np.all(np.isfinite(np.asarray(model.prototypes_)))

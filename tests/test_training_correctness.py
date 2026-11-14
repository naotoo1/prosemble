"""Regression tests for options and inputs that used to be handled wrongly
without any error.

Covers:
- a learning rate schedule keeping the chosen optimizer,
- training options that the scan loop does not implement,
- class labels that are not ``0..n_classes-1``,
- non-squared Euclidean distance when a prototype sits on a sample.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from sklearn.datasets import load_iris

from prosemble.core.distance import (
    euclidean_distance,
    euclidean_distance_matrix,
)
from prosemble.models import _MODEL_REGISTRY, GLVQ, OCGLVQ


@pytest.fixture(scope='module')
def iris():
    data = load_iris()
    return data.data.astype(np.float32), data.target.astype(np.int32)


# --- Learning rate schedule keeps the optimizer ---

class TestSchedulerKeepsOptimizer:

    @pytest.mark.parametrize('optimizer', ['adam', 'sgd', 'rmsprop', 'adamw'])
    def test_constant_schedule_matches_constant_lr(self, iris, optimizer):
        X, y = iris
        kwargs = dict(max_iter=20, epsilon=0.0, lr=0.01, optimizer=optimizer)
        plain = GLVQ(**kwargs).fit(X, y)
        scheduled = GLVQ(
            lr_scheduler=optax.constant_schedule(0.01), **kwargs
        ).fit(X, y)
        np.testing.assert_allclose(
            scheduled.prototypes_, plain.prototypes_, atol=1e-6
        )

    def test_scheduled_rmsprop_is_not_plain_sgd(self, iris):
        X, y = iris
        schedule = optax.constant_schedule(0.01)
        kwargs = dict(max_iter=20, epsilon=0.0, lr=0.01, lr_scheduler=schedule)
        rmsprop = GLVQ(optimizer='rmsprop', **kwargs).fit(X, y)
        sgd = GLVQ(optimizer='sgd', **kwargs).fit(X, y)
        assert not np.allclose(rmsprop.prototypes_, sgd.prototypes_, atol=1e-6)

    def test_named_schedule_changes_training(self, iris):
        X, y = iris
        kwargs = dict(max_iter=20, epsilon=0.0, lr=0.05, optimizer='adam')
        plain = GLVQ(**kwargs).fit(X, y)
        decayed = GLVQ(lr_scheduler='cosine_decay', **kwargs).fit(X, y)
        assert not np.allclose(plain.prototypes_, decayed.prototypes_)

    def test_prebuilt_optimizer_with_scheduler_raises(self):
        with pytest.raises(ValueError, match='pre-built optax optimizer'):
            GLVQ(optimizer=optax.adam(0.01), lr_scheduler='cosine_decay')

    @pytest.mark.parametrize('optimizer', ['sm3', 'rprop'])
    def test_constant_lr_only_optimizer_with_scheduler_raises(self, optimizer):
        with pytest.raises(ValueError, match='constant learning rate'):
            GLVQ(optimizer=optimizer, lr_scheduler='cosine_decay')

    @pytest.mark.parametrize('optimizer', [
        'adam', 'adamw', 'adamax', 'adamaxw', 'adan', 'adabelief', 'amsgrad',
        'radam', 'lamb', 'lion', 'novograd', 'sgd', 'sign_sgd', 'signum',
        'noisy_sgd', 'lars', 'rmsprop', 'adagrad', 'adadelta', 'adafactor',
        'yogi', 'fromage',
    ])
    def test_every_schedulable_optimizer_trains(self, iris, optimizer):
        X, y = iris
        model = GLVQ(
            max_iter=3, epsilon=0.0, optimizer=optimizer,
            lr_scheduler='cosine_decay',
        ).fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.prototypes_)))


# --- Options the scan loop does not implement ---

class TestOptionsHonouredWithDefaultScan:

    def test_sample_weight_changes_result(self, iris):
        X, y = iris
        weights = np.where(y == 0, 100.0, 0.001).astype(np.float32)
        plain = GLVQ(max_iter=20).fit(X, y)
        weighted = GLVQ(max_iter=20).fit(X, y, sample_weight=weights)
        assert not np.allclose(plain.prototypes_, weighted.prototypes_)

    def test_class_weight_changes_result(self, iris):
        X, y = iris
        plain = GLVQ(max_iter=20).fit(X, y)
        weighted = GLVQ(
            max_iter=20, class_weight={0: 100.0, 1: 0.001, 2: 0.001}
        ).fit(X, y)
        assert not np.allclose(plain.prototypes_, weighted.prototypes_)

    def test_validation_data_is_monitored(self, iris):
        X, y = iris
        model = GLVQ(max_iter=15).fit(X, y, validation_data=(X, y))
        assert model.val_loss_history_ is not None
        assert len(model.val_loss_history_) == model.n_iter_

    def test_restore_best_sets_best_loss(self, iris):
        X, y = iris
        model = GLVQ(max_iter=15, restore_best=True).fit(X, y)
        assert model.best_loss_ is not None
        assert model.best_loss_ == pytest.approx(
            float(jnp.min(model.loss_history_))
        )

    def test_patience_stops_early(self, iris):
        X, y = iris
        # A learning rate this small cannot improve the loss by more than
        # epsilon, so patience must stop training well before max_iter.
        model = GLVQ(max_iter=200, lr=1e-9, epsilon=1e-3, patience=3).fit(X, y)
        assert model.n_iter_ < 200
        assert len(model.loss_history_) == model.n_iter_

    def test_ema_changes_result(self, iris):
        X, y = iris
        plain = GLVQ(max_iter=20, epsilon=0.0).fit(X, y)
        ema = GLVQ(max_iter=20, epsilon=0.0, ema_decay=0.9).fit(X, y)
        assert not np.allclose(plain.prototypes_, ema.prototypes_)

    def test_minibatch_sample_weight_changes_result(self, iris):
        X, y = iris
        weights = np.where(y == 0, 100.0, 0.001).astype(np.float32)
        plain = GLVQ(max_iter=5, batch_size=32).fit(X, y)
        weighted = GLVQ(max_iter=5, batch_size=32).fit(
            X, y, sample_weight=weights
        )
        assert not np.allclose(plain.prototypes_, weighted.prototypes_)

    def test_scan_still_used_without_these_options(self, iris):
        X, y = iris
        model = GLVQ(max_iter=12).fit(X, y)
        # The scan loop always records max_iter losses.
        assert len(model.loss_history_) == 12


# --- Class labels that are not 0..n_classes-1 ---

LABEL_SETS = [[1, 2, 3], [0, 5, 9], [-1, 0, 1]]

ENCODED_MODELS = [
    'GLVQ', 'GMLVQ', 'GRLVQ', 'LGMLVQ', 'GTLVQ', 'CELVQ', 'CBC',
    'SLVQ', 'RSLVQ', 'MRSLVQ', 'LVQ1', 'LVQ21', 'MedianLVQ',
    'SRNG', 'CELVQ_NG',
]


class TestNonContiguousLabels:

    @pytest.mark.parametrize('name', ENCODED_MODELS)
    @pytest.mark.parametrize('labels', LABEL_SETS)
    def test_matches_contiguous_labels(self, iris, name, labels):
        """Relabelling the classes must only relabel the outputs."""
        X, y = iris
        cls = _MODEL_REGISTRY[name]
        labels = np.asarray(labels)

        reference = cls(max_iter=6).fit(X, y)
        model = cls(max_iter=6).fit(X, labels[y])

        np.testing.assert_array_equal(np.asarray(model.classes_), labels)
        assert model.n_classes_ == 3

        expected = labels[np.asarray(reference.predict(X))]
        np.testing.assert_array_equal(np.asarray(model.predict(X)), expected)

        if hasattr(model, 'predict_proba'):
            np.testing.assert_allclose(
                np.asarray(model.predict_proba(X)),
                np.asarray(reference.predict_proba(X)), atol=1e-5,
            )

    @pytest.mark.parametrize('labels', LABEL_SETS)
    def test_predict_proba_agrees_with_predict(self, iris, labels):
        X, y = iris
        labels = np.asarray(labels)
        model = GLVQ(max_iter=20).fit(X, labels[y])
        from_proba = np.asarray(model.classes_)[
            np.asarray(model.predict_proba(X)).argmax(axis=1)
        ]
        np.testing.assert_array_equal(from_proba, np.asarray(model.predict(X)))

    def test_contiguous_labels_are_untouched(self, iris):
        X, y = iris
        model = GLVQ(max_iter=6).fit(X, y)
        np.testing.assert_array_equal(np.asarray(model.classes_), [0, 1, 2])
        assert set(np.asarray(model.prototype_labels_).tolist()) == {0, 1, 2}
        assert set(np.asarray(model.predict(X)).tolist()) <= {0, 1, 2}

    def test_save_load_keeps_original_labels(self, iris, tmp_path):
        X, y = iris
        labels = np.asarray([0, 5, 9])
        model = GLVQ(max_iter=10).fit(X, labels[y])
        model.save(str(tmp_path / 'model'))
        loaded = GLVQ.load(str(tmp_path / 'model'))
        np.testing.assert_array_equal(np.asarray(loaded.classes_), labels)
        np.testing.assert_array_equal(
            np.asarray(loaded.predict(X)), np.asarray(model.predict(X))
        )
        np.testing.assert_allclose(
            np.asarray(loaded.predict_proba(X)),
            np.asarray(model.predict_proba(X)), atol=1e-6,
        )

    def test_file_saved_without_classes_still_loads(self, iris, tmp_path):
        """Files written before ``classes_`` was stored hold the original
        class values in ``prototype_labels_``."""
        import json

        X, y = iris
        labels = np.asarray([0, 5, 9])
        model = GLVQ(max_iter=10).fit(X, labels[y])
        expected = np.asarray(model.predict(X))
        model.save(str(tmp_path / 'new'))

        data = dict(np.load(str(tmp_path / 'new.npz'), allow_pickle=False))
        metadata = json.loads(str(data['__metadata__']))
        del metadata['classes_']
        data['__metadata__'] = np.array(json.dumps(metadata))
        data['prototype_labels_'] = labels[data['prototype_labels_']]
        np.savez_compressed(str(tmp_path / 'old'), **data)

        loaded = GLVQ.load(str(tmp_path / 'old'))
        np.testing.assert_array_equal(np.asarray(loaded.classes_), labels)
        np.testing.assert_array_equal(np.asarray(loaded.predict(X)), expected)

    def test_validation_and_partial_fit_accept_original_labels(self, iris):
        X, y = iris
        labels = np.asarray([1, 2, 3])
        model = GLVQ(max_iter=10).fit(
            X, labels[y], validation_data=(X, labels[y])
        )
        assert model.val_loss_history_ is not None
        before = np.asarray(model.prototypes_).copy()
        model.partial_fit(X, labels[y])
        assert not np.allclose(before, np.asarray(model.prototypes_))
        ratios = np.asarray(model.prototype_win_ratios(X, labels[y]))
        assert ratios.sum() == pytest.approx(
            float((np.asarray(model.predict(X)) == labels[y]).mean()), abs=1e-6
        )

    @pytest.mark.parametrize('name', [
        'SLVQ', 'RSLVQ', 'MRSLVQ', 'LMRSLVQ', 'RSLVQ_NG', 'MRSLVQ_NG',
        'LMRSLVQ_NG',
    ])
    def test_predict_with_rejection_returns_original_labels(self, iris, name):
        X, y = iris
        labels = np.asarray([0, 5, 9])
        model = _MODEL_REGISTRY[name](max_iter=6).fit(X, labels[y])
        np.testing.assert_array_equal(
            np.asarray(model.predict_with_rejection(X, confidence=0.0)),
            np.asarray(model.predict(X)),
        )
        rejected = np.asarray(model.predict_with_rejection(X, confidence=2.0))
        assert set(rejected.tolist()) == {-1}

    def test_unseen_label_raises(self, iris):
        X, y = iris
        model = GLVQ(max_iter=5).fit(X, np.asarray([1, 2, 3])[y])
        with pytest.raises(ValueError, match='not seen during fit'):
            model.partial_fit(X, y)

    def test_class_weight_uses_original_labels(self, iris):
        X, y = iris
        labels = np.asarray([0, 5, 9])
        reference = GLVQ(
            max_iter=10, class_weight={0: 100.0, 1: 0.001, 2: 0.001}
        ).fit(X, y)
        model = GLVQ(
            max_iter=10, class_weight={0: 100.0, 5: 0.001, 9: 0.001}
        ).fit(X, labels[y])
        np.testing.assert_allclose(
            model.prototypes_, reference.prototypes_, atol=1e-6
        )

    def test_exported_predict_returns_original_labels(self, iris):
        X, y = iris
        labels = np.asarray([0, 5, 9])
        model = GLVQ(max_iter=10).fit(X, labels[y])
        exported = model.export_predict(batch_size=len(X))
        np.testing.assert_array_equal(
            np.asarray(exported.call(jnp.asarray(X))),
            np.asarray(model.predict(X)),
        )

    def test_one_class_models_use_labels_as_given(self, iris):
        X, y = iris
        binary = (y == 0).astype(np.int32)
        model = OCGLVQ(max_iter=10, target_label=1).fit(X, binary)
        assert model._target_label == 1
        assert set(np.asarray(model.predict(X)).tolist()) <= {0, 1}


# --- Non-squared Euclidean distance at zero ---

class TestEuclideanDistanceAtZero:

    def test_matrix_gradient_is_finite_for_coincident_points(self):
        X = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
        grad = jax.grad(
            lambda P: jnp.sum(euclidean_distance_matrix(X, P))
        )(X[:1])
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_vector_gradient_is_finite_for_coincident_points(self):
        x = jnp.asarray([1.0, 2.0])
        grad = jax.grad(euclidean_distance)(x, x)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_values_unchanged(self):
        X = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
        Y = jnp.asarray([[0.0, 0.0], [3.0, 4.0]])
        D = np.asarray(euclidean_distance_matrix(X, Y))
        np.testing.assert_allclose(
            D, [[0.0, 5.0], [np.sqrt(2.0), np.sqrt(13.0)]], atol=1e-6
        )

    def test_glvq_trains_with_euclidean_distance(self, iris):
        X, y = iris
        model = GLVQ(
            max_iter=20, distance_fn=euclidean_distance_matrix
        ).fit(X, y)
        assert bool(jnp.all(jnp.isfinite(model.prototypes_)))
        assert np.isfinite(model.loss_)
        assert float((np.asarray(model.predict(X)) == y).mean()) > 0.8

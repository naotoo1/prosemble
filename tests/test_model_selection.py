"""Tests for clone, cross_val_score, and GridSearchCV."""

import pytest
import jax.numpy as jnp
import numpy as np

from prosemble.core.model_selection import clone, cross_val_score, GridSearchCV
from prosemble.core.pipeline import Pipeline, StandardScaler, PCA
from prosemble.core.pipeline import NotFittedError


@pytest.fixture(scope="module")
def iris_data():
    from prosemble.datasets import load_iris_jax
    ds = load_iris_jax()
    return ds.input_data, ds.labels


# --- clone ---

class TestClone:
    def test_clone_glvq(self):
        from prosemble.models import GLVQ
        model = GLVQ(n_prototypes_per_class=2, max_iter=50, lr=0.05)
        model2 = clone(model)
        assert isinstance(model2, GLVQ)
        assert model2.n_prototypes_per_class == 2
        assert model2.lr == 0.05
        assert model2 is not model

    def test_clone_fcm(self):
        from prosemble.models import FCM
        model = FCM(n_clusters=4, max_iter=80)
        model2 = clone(model)
        assert isinstance(model2, FCM)
        assert model2.n_clusters == 4
        assert model2.max_iter == 80

    def test_clone_with_override(self):
        from prosemble.models import GLVQ
        model = GLVQ(n_prototypes_per_class=1, lr=0.01)
        model2 = clone(model, lr=0.05)
        assert model2.lr == 0.05
        assert model.lr == 0.01  # original unchanged

    def test_clone_pipeline(self):
        from prosemble.models import GLVQ
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GLVQ(n_prototypes_per_class=1, max_iter=30)),
        ])
        pipe2 = clone(pipe)
        assert isinstance(pipe2, Pipeline)
        assert pipe2 is not pipe
        assert pipe2.steps[1][1] is not pipe.steps[1][1]

    def test_clone_no_get_params_raises(self):
        class Bare:
            pass
        with pytest.raises(TypeError, match="get_params"):
            clone(Bare())


# --- cross_val_score ---

class TestCrossValScore:
    def test_basic_supervised(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        scores = cross_val_score(
            GLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01),
            X, y, cv=3,
        )
        assert scores.shape == (3,)
        assert jnp.all(scores >= 0.0)
        assert jnp.all(scores <= 1.0)

    def test_returns_jax_array(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        scores = cross_val_score(
            GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.01),
            X, y, cv=3,
        )
        assert isinstance(scores, jnp.ndarray)

    def test_different_cv_values(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.01)
        scores_3 = cross_val_score(model, X, y, cv=3)
        scores_5 = cross_val_score(model, X, y, cv=5)
        assert scores_3.shape == (3,)
        assert scores_5.shape == (5,)

    def test_custom_scorer(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data

        def my_scorer(y_true, y_pred):
            return float(jnp.mean(y_true == y_pred))

        scores = cross_val_score(
            GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.01),
            X, y, cv=3, scoring=my_scorer,
        )
        assert scores.shape == (3,)

    def test_reproducible(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        model = GLVQ(n_prototypes_per_class=1, max_iter=20, lr=0.01)
        scores1 = cross_val_score(model, X, y, cv=3, random_seed=42)
        scores2 = cross_val_score(model, X, y, cv=3, random_seed=42)
        np.testing.assert_allclose(scores1, scores2, atol=1e-5)


# --- GridSearchCV ---

class TestGridSearchCV:
    def test_basic_supervised(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        gs = GridSearchCV(
            GLVQ(max_iter=20),
            {'n_prototypes_per_class': [1, 2], 'lr': [0.01, 0.05]},
            cv=3,
        )
        gs.fit(X, y)
        assert gs.best_params_ is not None
        assert gs.best_score_ > 0.0

    def test_best_estimator_predicts(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        gs = GridSearchCV(
            GLVQ(max_iter=20),
            {'lr': [0.01, 0.05]},
            cv=3,
        )
        gs.fit(X, y)
        preds = gs.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_cv_results_structure(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        gs = GridSearchCV(
            GLVQ(max_iter=20),
            {'lr': [0.01, 0.05]},
            cv=3,
        )
        gs.fit(X, y)
        assert 'params' in gs.cv_results_
        assert 'mean_score' in gs.cv_results_
        assert 'std_score' in gs.cv_results_
        assert 'fold_scores' in gs.cv_results_
        assert 'rank' in gs.cv_results_
        assert len(gs.cv_results_['params']) == 2

    def test_ranking(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        gs = GridSearchCV(
            GLVQ(max_iter=20),
            {'lr': [0.01, 0.05]},
            cv=3,
        )
        gs.fit(X, y)
        ranks = gs.cv_results_['rank']
        assert 1 in ranks
        assert 2 in ranks

    def test_refit_false(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        gs = GridSearchCV(
            GLVQ(max_iter=20),
            {'lr': [0.01, 0.05]},
            cv=3,
            refit=False,
        )
        gs.fit(X, y)
        assert gs.best_params_ is not None
        assert gs.best_estimator_ is None
        with pytest.raises(NotFittedError):
            gs.predict(X)

    def test_pipeline_grid_search(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GLVQ(n_prototypes_per_class=1, max_iter=20)),
        ])
        gs = GridSearchCV(
            pipe,
            {'model__lr': [0.01, 0.05]},
            cv=3,
        )
        gs.fit(X, y)
        assert gs.best_params_ is not None
        preds = gs.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_single_param_value(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        gs = GridSearchCV(
            GLVQ(max_iter=20),
            {'lr': [0.01]},
            cv=3,
        )
        gs.fit(X, y)
        assert gs.best_params_ == {'lr': 0.01}

    def test_predict_before_fit_raises(self):
        from prosemble.models import GLVQ
        gs = GridSearchCV(GLVQ(), {'lr': [0.01]}, cv=3)
        with pytest.raises(NotFittedError):
            gs.predict(jnp.array([[1.0, 2.0]]))

    def test_repr(self):
        from prosemble.models import GLVQ
        gs = GridSearchCV(GLVQ(), {'lr': [0.01, 0.05]}, cv=3)
        r = repr(gs)
        assert 'GridSearchCV' in r
        assert 'GLVQ' in r

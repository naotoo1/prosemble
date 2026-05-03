"""Tests for Pipeline and Transformers."""

import pytest
import jax.numpy as jnp
import numpy as np

from prosemble.core.pipeline import (
    Pipeline, StandardScaler, MinMaxScaler, PCA, TransformerMixin,
)
from prosemble.core.pipeline import NotFittedError


@pytest.fixture(scope="module")
def iris_data():
    from prosemble.datasets import load_iris_jax
    ds = load_iris_jax()
    return ds.input_data, ds.labels


@pytest.fixture
def simple_data():
    return jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ])


# --- StandardScaler ---

class TestStandardScaler:
    def test_fit_transform_zero_mean(self, simple_data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(simple_data)
        np.testing.assert_allclose(
            jnp.mean(X_scaled, axis=0), 0.0, atol=1e-5
        )

    def test_fit_transform_unit_var(self, simple_data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(simple_data)
        np.testing.assert_allclose(
            jnp.std(X_scaled, axis=0), 1.0, atol=1e-5
        )

    def test_transform_uses_training_stats(self, simple_data):
        scaler = StandardScaler()
        scaler.fit(simple_data)
        X_new = jnp.array([[1.0, 2.0, 3.0]])
        X_scaled = scaler.transform(X_new)
        assert X_scaled.shape == (1, 3)
        # Should use training mean/std, not recompute
        assert not jnp.allclose(X_scaled, jnp.zeros(3))

    def test_not_fitted_raises(self):
        scaler = StandardScaler()
        with pytest.raises(NotFittedError):
            scaler.transform(jnp.array([[1.0, 2.0]]))

    def test_constant_feature_no_nan(self):
        X = jnp.array([[1.0, 5.0], [1.0, 3.0], [1.0, 7.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert jnp.all(jnp.isfinite(X_scaled))

    def test_output_is_jax_array(self, simple_data):
        scaler = StandardScaler()
        result = scaler.fit_transform(simple_data)
        assert isinstance(result, jnp.ndarray)


# --- MinMaxScaler ---

class TestMinMaxScaler:
    def test_fit_transform_range_01(self, simple_data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(simple_data)
        np.testing.assert_allclose(jnp.min(X_scaled, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(jnp.max(X_scaled, axis=0), 1.0, atol=1e-5)

    def test_transform_uses_training_range(self, simple_data):
        scaler = MinMaxScaler()
        scaler.fit(simple_data)
        X_new = jnp.array([[13.0, 14.0, 15.0]])
        X_scaled = scaler.transform(X_new)
        # Values above training max should be > 1
        assert jnp.all(X_scaled > 1.0)

    def test_not_fitted_raises(self):
        scaler = MinMaxScaler()
        with pytest.raises(NotFittedError):
            scaler.transform(jnp.array([[1.0, 2.0]]))

    def test_output_is_jax_array(self, simple_data):
        scaler = MinMaxScaler()
        result = scaler.fit_transform(simple_data)
        assert isinstance(result, jnp.ndarray)


# --- PCA ---

class TestPCA:
    def test_dimensionality_reduction(self, iris_data):
        X, _ = iris_data
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape == (X.shape[0], 2)

    def test_transform_consistency(self, iris_data):
        X, _ = iris_data
        pca = PCA(n_components=2)
        X_ft = pca.fit_transform(X)
        X_t = pca.transform(X)
        np.testing.assert_allclose(X_ft, X_t, atol=1e-5)

    def test_not_fitted_raises(self):
        pca = PCA(n_components=2)
        with pytest.raises(NotFittedError):
            pca.transform(jnp.array([[1.0, 2.0, 3.0]]))

    def test_get_set_params(self):
        pca = PCA(n_components=3)
        assert pca.get_params() == {'n_components': 3}
        pca.set_params(n_components=2)
        assert pca.n_components == 2

    def test_output_is_jax_array(self, iris_data):
        X, _ = iris_data
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        assert isinstance(result, jnp.ndarray)


# --- Pipeline ---

class TestPipeline:
    def test_scaler_plus_glvq(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (X.shape[0],)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.5

    def test_scaler_pca_glvq(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('model', GLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_proba(self, iris_data):
        from prosemble.models import GLVQ
        X, y = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GLVQ(n_prototypes_per_class=1, max_iter=30, lr=0.01)),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        np.testing.assert_allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-4)

    def test_unsupervised_pipeline(self, iris_data):
        from prosemble.models import FCM
        X, _ = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', FCM(n_clusters=3, max_iter=50)),
        ])
        pipe.fit(X)
        preds = pipe.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_transform_all_steps(self, iris_data):
        X, _ = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2)),
        ])
        pipe.fit(X)
        X_t = pipe.transform(X)
        assert X_t.shape == (X.shape[0], 2)

    def test_get_params_deep(self):
        pca = PCA(n_components=3)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
        ])
        params = pipe.get_params(deep=True)
        assert 'pca__n_components' in params
        assert params['pca__n_components'] == 3

    def test_set_params_nested(self):
        pca = PCA(n_components=3)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
        ])
        pipe.set_params(pca__n_components=2)
        assert pca.n_components == 2

    def test_duplicate_step_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Pipeline([
                ('a', StandardScaler()),
                ('a', MinMaxScaler()),
            ])

    def test_intermediate_without_transform_raises(self):
        class NoTransform:
            def fit(self, X, y=None): return self
        with pytest.raises(TypeError, match="transform"):
            Pipeline([
                ('bad', NoTransform()),
                ('scaler', StandardScaler()),
            ])

    def test_jax_arrays_throughout(self, iris_data):
        X, _ = iris_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2)),
        ])
        pipe.fit(X)
        result = pipe.transform(X)
        assert isinstance(result, jnp.ndarray)

    def test_repr(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2)),
        ])
        r = repr(pipe)
        assert 'Pipeline' in r
        assert 'scaler' in r
        assert 'pca' in r

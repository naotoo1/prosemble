"""JAX-native Pipeline and Transformers for prosemble.

Provides sklearn-compatible Pipeline, StandardScaler, MinMaxScaler, and PCA
using pure JAX operations — no numpy roundtrips.
"""

import inspect
from typing import Self

import jax.numpy as jnp
import chex

from prosemble.core.utils import standardize, min_max_scale, pca_jax


class NotFittedError(ValueError, RuntimeError):
    """Raised when calling transform/predict on an unfitted estimator."""


class TransformerMixin:
    """Mixin providing fit_transform."""

    def fit_transform(self, X: chex.Array, y: chex.Array | None = None) -> chex.Array:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class StandardScaler(TransformerMixin):
    """Standardize features to zero mean and unit variance.

    Wraps ``prosemble.core.utils.standardize()``.

    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None) -> Self:
        """Compute mean and std from training data."""
        X = jnp.asarray(X, dtype=jnp.float32)
        _, self.mean_, self.std_ = standardize(X)
        return self

    def transform(self, X) -> chex.Array:
        """Standardize using stored mean and std."""
        if self.mean_ is None:
            raise NotFittedError("StandardScaler not fitted. Call fit() first.")
        X = jnp.asarray(X, dtype=jnp.float32)
        result, _, _ = standardize(X, mean=self.mean_, std=self.std_)
        return result

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class MinMaxScaler(TransformerMixin):
    """Scale features to [0, 1] range.

    Wraps ``prosemble.core.utils.min_max_scale()``.

    Examples
    --------
    >>> scaler = MinMaxScaler()
    >>> X_scaled = scaler.fit_transform(X_train)
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None) -> Self:
        """Compute min and max from training data."""
        X = jnp.asarray(X, dtype=jnp.float32)
        _, self.min_, self.max_ = min_max_scale(X)
        return self

    def transform(self, X) -> chex.Array:
        """Scale using stored min and max."""
        if self.min_ is None:
            raise NotFittedError("MinMaxScaler not fitted. Call fit() first.")
        X = jnp.asarray(X, dtype=jnp.float32)
        result, _, _ = min_max_scale(X, min_val=self.min_, max_val=self.max_)
        return result

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class PCA(TransformerMixin):
    """Principal Component Analysis.

    Wraps ``prosemble.core.utils.pca_jax()``.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to keep.

    Examples
    --------
    >>> pca = PCA(n_components=2)
    >>> X_reduced = pca.fit_transform(X)
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X, y=None) -> Self:
        """Compute principal components from training data."""
        X = jnp.asarray(X, dtype=jnp.float32)
        self.mean_ = jnp.mean(X, axis=0)
        _, self.components_ = pca_jax(X, n_components=self.n_components)
        return self

    def transform(self, X) -> chex.Array:
        """Project data onto principal components."""
        if self.components_ is None:
            raise NotFittedError("PCA not fitted. Call fit() first.")
        X = jnp.asarray(X, dtype=jnp.float32)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def get_params(self, deep=True):
        return {'n_components': self.n_components}

    def set_params(self, **params):
        for key, val in params.items():
            if key == 'n_components':
                self.n_components = val
            else:
                raise ValueError(f"Invalid parameter '{key}' for PCA.")
        return self


class Pipeline:
    """Chain transformers with a final estimator.

    All operations use pure JAX arrays — no numpy roundtrips.

    Parameters
    ----------
    steps : list of (name, estimator) tuples
        Sequence of transforms with a final estimator.
        All but the last must implement ``fit()`` and ``transform()``.
        The last step can be any estimator or transformer.

    Examples
    --------
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('pca', PCA(n_components=2)),
    ...     ('model', GLVQ(n_prototypes_per_class=1, max_iter=50)),
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> preds = pipe.predict(X_test)
    """

    def __init__(self, steps: list[tuple[str, object]]):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        """Check step format and uniqueness of names."""
        names = []
        for i, (name, est) in enumerate(self.steps):
            if not isinstance(name, str):
                raise TypeError(f"Step name must be str, got {type(name)}")
            if name in names:
                raise ValueError(f"Duplicate step name: '{name}'")
            names.append(name)
            if i < len(self.steps) - 1:
                if not hasattr(est, 'transform'):
                    raise TypeError(
                        f"Intermediate step '{name}' must implement transform(). "
                        f"Got {type(est).__name__}"
                    )

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    def _transform_steps(self, X, fit=False, y=None):
        """Apply all intermediate steps (all but last)."""
        Xt = jnp.asarray(X, dtype=jnp.float32)
        for name, transformer in self.steps[:-1]:
            if fit:
                if hasattr(transformer, 'fit_transform'):
                    Xt = transformer.fit_transform(Xt, y)
                else:
                    transformer.fit(Xt, y)
                    Xt = transformer.transform(Xt)
            else:
                Xt = transformer.transform(Xt)
        return Xt

    def fit(self, X, y=None, **fit_params) -> Self:
        """Fit all steps.

        Calls fit_transform on intermediate steps, fit on the final step.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,), optional
        **fit_params : forwarded to the final estimator's fit().
        """
        Xt = self._transform_steps(X, fit=True, y=y)
        final_name, final_est = self.steps[-1]
        sig = inspect.signature(final_est.fit)
        if 'y' in sig.parameters and y is not None:
            final_est.fit(Xt, y, **fit_params)
        else:
            final_est.fit(Xt, **fit_params)
        return self

    def predict(self, X):
        """Transform X through pipeline, then predict."""
        Xt = self._transform_steps(X, fit=False)
        return self._final_estimator.predict(Xt)

    def predict_proba(self, X):
        """Transform X through pipeline, then predict_proba."""
        Xt = self._transform_steps(X, fit=False)
        if not hasattr(self._final_estimator, 'predict_proba'):
            raise AttributeError(
                f"Final estimator {type(self._final_estimator).__name__} "
                f"has no predict_proba method."
            )
        return self._final_estimator.predict_proba(Xt)

    def transform(self, X):
        """Transform X through all steps (including last if it has transform)."""
        Xt = self._transform_steps(X, fit=False)
        if hasattr(self._final_estimator, 'transform'):
            return self._final_estimator.transform(Xt)
        raise AttributeError(
            f"Final estimator {type(self._final_estimator).__name__} "
            f"has no transform method."
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_params(self, deep=True):
        """Get pipeline parameters.

        If deep=True, includes nested estimator params as ``name__param``.
        """
        params = {'steps': self.steps}
        if deep:
            for name, est in self.steps:
                if hasattr(est, 'get_params'):
                    for key, val in est.get_params().items():
                        params[f'{name}__{key}'] = val
        return params

    def set_params(self, **params):
        """Set pipeline parameters.

        Supports nested params: ``model__lr=0.05`` sets lr on step 'model'.
        """
        step_dict = dict(self.steps)
        for key, val in params.items():
            if '__' in key:
                step_name, param_name = key.split('__', 1)
                if step_name not in step_dict:
                    raise ValueError(f"No step named '{step_name}'")
                step_dict[step_name].set_params(**{param_name: val})
            elif key == 'steps':
                self.steps = val
                self._validate_steps()
            else:
                raise ValueError(f"Invalid parameter '{key}' for Pipeline")
        return self

    def __repr__(self):
        step_reprs = ', '.join(f"('{n}', {type(e).__name__})" for n, e in self.steps)
        return f"Pipeline([{step_reprs}])"

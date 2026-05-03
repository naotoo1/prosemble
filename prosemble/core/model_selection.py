"""JAX-native model selection: clone, cross_val_score, GridSearchCV.

Pure JAX implementation — no sklearn dependency, no numpy roundtrips.
"""

import inspect
import itertools
from typing import Callable, Self

import jax.numpy as jnp
import chex

from prosemble.core.utils import k_fold_split_jax, accuracy_score_jax
from prosemble.core.pipeline import NotFittedError


def clone(estimator, **override_params):
    """Create a fresh (unfitted) copy of an estimator.

    Parameters
    ----------
    estimator : object
        Estimator with ``get_params()`` protocol.
    **override_params
        Parameters to override in the clone.

    Returns
    -------
    new_estimator : same type as estimator
        Fresh, unfitted instance.

    Examples
    --------
    >>> model = GLVQ(n_prototypes_per_class=2, lr=0.01)
    >>> model2 = clone(model, lr=0.05)
    """
    from prosemble.core.pipeline import Pipeline

    if isinstance(estimator, Pipeline):
        return _clone_pipeline(estimator, **override_params)

    if not hasattr(estimator, 'get_params'):
        raise TypeError(
            f"Cannot clone {type(estimator).__name__}: no get_params() method."
        )

    params = estimator.get_params(deep=False)
    params.update(override_params)

    # Filter to params accepted by __init__ (including **kwargs)
    init_sig = inspect.signature(type(estimator).__init__)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in init_sig.parameters.values()
    )

    if has_var_keyword:
        # Constructor accepts **kwargs, pass all params
        valid_params = {k: v for k, v in params.items() if k != 'self'}
    else:
        valid_params = {
            k: v for k, v in params.items()
            if k in init_sig.parameters and k != 'self'
        }

    return type(estimator)(**valid_params)


def _clone_pipeline(pipeline, **override_params):
    """Deep-clone a Pipeline, cloning each step's estimator."""
    from prosemble.core.pipeline import Pipeline

    new_steps = []
    for name, est in pipeline.steps:
        new_steps.append((name, clone(est)))
    new_pipe = Pipeline(new_steps)
    if override_params:
        new_pipe.set_params(**override_params)
    return new_pipe


def _is_supervised(estimator):
    """Check if estimator's fit() accepts y parameter."""
    sig = inspect.signature(estimator.fit)
    return 'y' in sig.parameters


def _get_scorer(scoring):
    """Resolve scoring to a callable."""
    if scoring == 'accuracy':
        return accuracy_score_jax
    elif callable(scoring):
        return scoring
    else:
        raise ValueError(
            f"Unknown scoring '{scoring}'. Use 'accuracy' or a callable."
        )


def cross_val_score(
    estimator,
    X: chex.Array,
    y: chex.Array | None = None,
    cv: int = 5,
    scoring: str | Callable = 'accuracy',
    random_seed: int = 42,
) -> chex.Array:
    """Evaluate estimator with cross-validation.

    Parameters
    ----------
    estimator : object
        Estimator with fit/predict and get_params.
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,), optional
        Required for supervised estimators.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str or callable, default='accuracy'
        'accuracy' or callable ``scorer(y_true, y_pred) -> float``.
        For unsupervised without y: ``scorer(estimator, X_test) -> float``.
    random_seed : int, default=42

    Returns
    -------
    scores : jnp.ndarray of shape (cv,)
        Score for each fold.

    Examples
    --------
    >>> scores = cross_val_score(GLVQ(max_iter=30), X, y, cv=5)
    >>> print(f"Mean: {scores.mean():.3f} +/- {scores.std():.3f}")
    """
    X = jnp.asarray(X)
    if y is not None:
        y = jnp.asarray(y)

    scorer = _get_scorer(scoring)
    supervised = _is_supervised(estimator)
    fold_scores = []

    for train_idx, test_idx in k_fold_split_jax(X.shape[0], cv, random_seed):
        est = clone(estimator)
        X_train, X_test = X[train_idx], X[test_idx]

        if supervised:
            if y is None:
                raise ValueError("y must be provided for supervised estimators.")
            y_train, y_test = y[train_idx], y[test_idx]
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            score = float(scorer(y_test, y_pred))
        else:
            est.fit(X_train)
            if y is not None:
                y_test = y[test_idx]
                y_pred = est.predict(X_test)
                score = float(scorer(y_test, y_pred))
            else:
                # Unsupervised custom scorer: scorer(estimator, X_test)
                score = float(scorer(est, X_test))

        fold_scores.append(score)

    return jnp.array(fold_scores)


class GridSearchCV:
    """Exhaustive search over a parameter grid with cross-validation.

    Parameters
    ----------
    estimator : object
        Base estimator with fit/predict and get_params/set_params.
        Can be a Pipeline or any prosemble model.
    param_grid : dict
        Maps parameter names to lists of values to try.
        For Pipeline steps, use ``step_name__param`` notation.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str or callable, default='accuracy'
        'accuracy' or callable ``scorer(y_true, y_pred) -> float``.
    random_seed : int, default=42
    refit : bool, default=True
        If True, refit the best model on the full dataset after search.
    verbose : int, default=0
        0=silent, 1=per-combo summary, 2=per-fold detail.

    Attributes
    ----------
    best_params_ : dict
        Parameters of the best model.
    best_score_ : float
        Mean CV score of the best model.
    best_estimator_ : object
        Fitted estimator with best params (only if refit=True).
    cv_results_ : dict
        Keys: 'params', 'mean_score', 'std_score', 'fold_scores', 'rank'.

    Examples
    --------
    >>> gs = GridSearchCV(
    ...     GLVQ(max_iter=30),
    ...     {'n_prototypes_per_class': [1, 2], 'lr': [0.01, 0.05]},
    ...     cv=3,
    ... )
    >>> gs.fit(X, y)
    >>> print(gs.best_params_, gs.best_score_)
    """

    def __init__(
        self,
        estimator,
        param_grid: dict,
        cv: int = 5,
        scoring: str | Callable = 'accuracy',
        random_seed: int = 42,
        refit: bool = True,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.random_seed = random_seed
        self.refit = refit
        self.verbose = verbose

        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def _generate_param_combinations(self):
        """Generate all combinations from param_grid."""
        keys = sorted(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        return [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    def _clone_with_params(self, params):
        """Clone base estimator with overridden params."""
        from prosemble.core.pipeline import Pipeline
        if isinstance(self.estimator, Pipeline):
            return _clone_pipeline(self.estimator, **params)
        return clone(self.estimator, **params)

    def fit(self, X, y=None) -> Self:
        """Run grid search with cross-validation.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,), optional

        Returns
        -------
        self
        """
        X = jnp.asarray(X)
        if y is not None:
            y = jnp.asarray(y)

        param_combos = self._generate_param_combinations()
        all_results = []

        for combo_idx, params in enumerate(param_combos):
            est = self._clone_with_params(params)
            scores = cross_val_score(
                est, X, y,
                cv=self.cv,
                scoring=self.scoring,
                random_seed=self.random_seed,
            )

            mean_score = float(jnp.mean(scores))
            std_score = float(jnp.std(scores))

            all_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': scores.tolist(),
            })

            if self.verbose >= 1:
                print(
                    f"[{combo_idx + 1}/{len(param_combos)}] "
                    f"{params} -> {mean_score:.4f} +/- {std_score:.4f}"
                )

        # Rank by mean score (descending)
        sorted_indices = sorted(
            range(len(all_results)),
            key=lambda i: all_results[i]['mean_score'],
            reverse=True,
        )
        ranks = [0] * len(all_results)
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[idx] = rank

        self.cv_results_ = {
            'params': [r['params'] for r in all_results],
            'mean_score': [r['mean_score'] for r in all_results],
            'std_score': [r['std_score'] for r in all_results],
            'fold_scores': [r['fold_scores'] for r in all_results],
            'rank': ranks,
        }

        best_idx = sorted_indices[0]
        self.best_params_ = all_results[best_idx]['params']
        self.best_score_ = all_results[best_idx]['mean_score']

        if self.refit:
            self.best_estimator_ = self._clone_with_params(self.best_params_)
            if _is_supervised(self.best_estimator_):
                self.best_estimator_.fit(X, y)
            else:
                self.best_estimator_.fit(X)

        return self

    def predict(self, X):
        """Predict using best estimator."""
        if self.best_estimator_ is None:
            raise NotFittedError(
                "GridSearchCV not fitted or refit=False. "
                "Call fit() with refit=True first."
            )
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using best estimator."""
        if self.best_estimator_ is None:
            raise NotFittedError(
                "GridSearchCV not fitted or refit=False."
            )
        return self.best_estimator_.predict_proba(X)

    def __repr__(self):
        return (
            f"GridSearchCV(estimator={type(self.estimator).__name__}, "
            f"param_grid={self.param_grid}, cv={self.cv})"
        )

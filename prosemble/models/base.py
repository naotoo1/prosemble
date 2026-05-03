"""Base class for unsupervised fuzzy/possibilistic clustering models."""

import json
from abc import ABC, abstractmethod

from prosemble.core.quantization import MetadataCollectorMixin, QuantizationMixin
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


class NotFittedError(ValueError, RuntimeError):
    """Raised when a method is called on an unfitted model.

    Inherits from both ValueError and RuntimeError for backward
    compatibility with existing code that catches either type.
    """
    pass


class FuzzyClusteringBase(MetadataCollectorMixin, QuantizationMixin, ABC):
    """Base class providing shared boilerplate for clustering models.

    Provides common parameter handling, input validation, fitted-state
    checks, and accessor methods. Subclasses implement their own
    fit(), predict(), and predict_proba() with model-specific math.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (must be >= 2)
    max_iter : int, default=100
        Maximum number of iterations
    epsilon : float, default=1e-5
        Convergence threshold
    random_seed : int, default=42
        Random seed for reproducibility
    plot_steps : bool, default=False
        Whether to visualize clustering progress
    show_confidence : bool, default=True
        Whether to show confidence in visualization
    show_pca_variance : bool, default=True
        Whether to show PCA variance in visualization
    save_plot_path : str, optional
        Path to save final plot

    Class Attributes (for subclasses)
    ----------------------------------
    _hyperparams : tuple[str, ...]
        Names of model-specific hyperparameters (collected from MRO).
    _fitted_array_names : tuple[str, ...]
        Names of model-specific fitted arrays (collected from MRO).
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        random_seed: int = 42,
        distance_fn=None,
        patience: int = None,
        restore_best: bool = False,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str = None,
        callbacks: list = None,
    ):
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.patience = patience
        self.restore_best = restore_best
        self.key = jax.random.PRNGKey(random_seed)
        # Distance function (default: squared Euclidean)
        if distance_fn is None:
            from prosemble.core.distance import squared_euclidean_distance_matrix
            distance_fn = squared_euclidean_distance_matrix
        self.distance_fn = distance_fn

        self.plot_steps = plot_steps
        self.show_confidence = show_confidence
        self.show_pca_variance = show_pca_variance
        self.save_plot_path = save_plot_path

        # Callback system
        self._callbacks = list(callbacks or [])
        if plot_steps:
            from prosemble.core.callbacks import VisualizationCallback
            self._callbacks.append(VisualizationCallback(
                show_confidence=show_confidence,
                show_pca_variance=show_pca_variance,
                save_path=save_plot_path,
            ))

        # Common fitted attributes (set by fit())
        self.centroids_ = None
        self.n_iter_ = None
        self.objective_ = None
        self.objective_history_ = None
        self.best_loss_ = None

    _base_repr_fields = ('n_clusters', 'max_iter', 'epsilon', 'random_seed')

    def __repr__(self) -> str:
        cls = type(self).__name__
        base = ', '.join(f'{k}={getattr(self, k)!r}' for k in self._base_repr_fields)
        extra = ', '.join(f'{k}={getattr(self, k)!r}' for k in self._all_hyperparams)
        parts = f'{base}, {extra}' if extra else base
        return f'{cls}({parts})'

    def _validate_input(self, X):
        """Validate and convert input data to JAX array.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns
        -------
        X : jnp.ndarray
            Validated JAX array

        Raises
        ------
        ValueError
            If X is not 2D or has fewer samples than clusters
        """
        X = jnp.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        return X

    def _check_fitted(self, attr='centroids_'):
        """Raise NotFittedError if model has not been fitted.

        Parameters
        ----------
        attr : str
            Attribute name to check for None
        """
        if getattr(self, attr, None) is None:
            raise NotFittedError(
                "Model not fitted yet. Call fit() first."
            )

    # --- Abstract interface ---

    @abstractmethod
    def _iteration_step(self, state, X) -> tuple:
        """Single iteration of the clustering algorithm.

        Must return (new_state, metrics_dict).
        """
        ...

    @abstractmethod
    def predict(self, X):
        """Predict cluster labels for new data."""
        ...

    @abstractmethod
    def _build_info(self, state, iteration) -> dict:
        """Build info dict for callbacks/visualization."""
        ...

    def _get_quantizable_attrs(self) -> list[str]:
        """Return list of attr names for quantizable parameters."""
        attrs = []
        if self.centroids_ is not None:
            attrs.append('centroids_')
        return attrs

    def get_objective_history(self):
        """Return objective function values across iterations.

        Returns
        -------
        objectives : jnp.ndarray
            Objective values at each iteration
        """
        self._check_fitted('objective_history_')
        return self.objective_history_

    def final_centroids(self):
        """Return learned cluster centroids.

        Returns
        -------
        centroids : jnp.ndarray, shape (n_clusters, n_features)
            Cluster centroids
        """
        self._check_fitted()
        return self.centroids_

    def _validate_initial_centroids(self, X, initial_centroids):
        """Validate user-provided initial centroids.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_samples, n_features)
            Input data (used for shape check)
        initial_centroids : array-like
            User-provided centroids

        Returns
        -------
        initial_centroids : jnp.ndarray, shape (n_clusters, n_features)
        """
        initial_centroids = jnp.asarray(initial_centroids)
        expected_shape = (self.n_clusters, X.shape[1])
        if initial_centroids.shape != expected_shape:
            raise ValueError(
                f"initial_centroids shape {initial_centroids.shape} "
                f"does not match expected {expected_shape}"
            )
        return initial_centroids

    # --- Serialization (auto-collected from _hyperparams / _fitted_array_names) ---

    def _get_hyperparams(self) -> dict:
        """Return constructor hyperparameters as a dict of scalars/strings."""
        params = {
            'n_clusters': int(self.n_clusters),
            'max_iter': int(self.max_iter),
            'epsilon': float(self.epsilon),
            'random_seed': int(self.random_seed),
        }
        for name in self._all_hyperparams:
            val = getattr(self, name)
            params[name] = float(val) if hasattr(val, 'item') else val
        return params

    def _get_fitted_arrays(self) -> dict:
        """Return fitted arrays as a dict of numpy arrays."""
        arrays = {}
        for attr in ['centroids_', 'objective_history_']:
            val = getattr(self, attr, None)
            if val is not None:
                arrays[attr] = np.asarray(val)
        for name in self._all_fitted_array_names:
            val = getattr(self, name, None)
            if val is not None:
                arrays[name] = np.asarray(val)
        return arrays

    def _set_fitted_arrays(self, arrays: dict) -> None:
        """Restore fitted arrays from numpy dict."""
        for attr in ['centroids_', 'objective_history_']:
            if attr in arrays:
                setattr(self, attr, jnp.asarray(arrays[attr]))
        for name in self._all_fitted_array_names:
            if name in arrays:
                setattr(self, name, jnp.asarray(arrays[name]))

    def save(self, path, quantize=None):
        """Save fitted model to an NPZ file.

        Parameters
        ----------
        path : str
            File path (`.npz` extension added if not present)
        quantize : str, optional
            Quantize before saving: 'float16', 'bfloat16', or 'int8'.
            Model in memory is unchanged; only the saved file is quantized.
        """
        self._check_fitted()

        # Save originals and temporarily quantize if needed
        originals = {}
        if quantize is not None and not self.is_quantized:
            for attr in self._get_quantizable_attrs():
                originals[attr] = getattr(self, attr)
            self.quantize(quantize)

        arrays = self._get_fitted_arrays()
        hyperparams = self._get_hyperparams()

        metadata = {
            'class_name': type(self).__name__,
            'module': type(self).__module__,
            'hyperparams': hyperparams,
            'fitted_array_names': list(arrays.keys()),
            'n_iter_': int(self.n_iter_) if self.n_iter_ is not None else None,
            'objective_': float(self.objective_) if self.objective_ is not None else None,
            'quantized_dtype': self.quantized_dtype,
        }

        if self.quantized_dtype == 'int8' and hasattr(self, '_int8_scales'):
            for attr, scale in self._int8_scales.items():
                arrays[f'__scale__{attr}'] = np.asarray(scale)
            metadata['int8_scale_keys'] = list(self._int8_scales.keys())

        save_dict = {'__metadata__': np.array(json.dumps(metadata))}
        save_dict.update(arrays)

        np.savez_compressed(path, **save_dict)

        # Restore originals exactly (no precision loss)
        if originals:
            for attr, val in originals.items():
                setattr(self, attr, val)
            self._quantized_dtype = None
            if hasattr(self, '_int8_scales'):
                self._int8_scales = {}

    @classmethod
    def load(cls, path):
        """Load a fitted model from an NPZ file.

        Parameters
        ----------
        path : str
            Path to the `.npz` file

        Returns
        -------
        model : FuzzyClusteringBase
            Reconstructed fitted model
        """
        from prosemble.models import _MODEL_REGISTRY

        if not path.endswith('.npz'):
            path = path + '.npz'

        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data['__metadata__']))

        class_name = metadata['class_name']
        if class_name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model class: {class_name}")

        model_cls = _MODEL_REGISTRY[class_name]
        hyperparams = metadata['hyperparams']

        model = model_cls(**hyperparams)

        arrays = {name: data[name] for name in metadata['fitted_array_names']}
        model._set_fitted_arrays(arrays)

        if metadata.get('n_iter_') is not None:
            model.n_iter_ = metadata['n_iter_']
        if metadata.get('objective_') is not None:
            model.objective_ = metadata['objective_']

        q_dtype = metadata.get('quantized_dtype')
        if q_dtype:
            model._quantized_dtype = q_dtype
            if q_dtype == 'int8':
                model._int8_scales = {}
                for attr in metadata.get('int8_scale_keys', []):
                    scale_key = f'__scale__{attr}'
                    if scale_key in data:
                        model._int8_scales[attr] = jnp.asarray(data[scale_key])

        return model

    # --- Patience-based early stopping ---

    def _check_patience(self, objective_history, patience):
        """Check if training should stop due to lack of improvement.

        Returns True if the last `patience` objectives did not improve
        over the best objective seen so far.

        Parameters
        ----------
        objective_history : list of float
            Objective values collected so far
        patience : int
            Number of consecutive non-improving epochs to tolerate

        Returns
        -------
        should_stop : bool
        """
        if len(objective_history) <= patience:
            return False
        best = min(objective_history[:-patience])
        recent = objective_history[-patience:]
        return all(r >= best for r in recent)

    # --- Callback notifications ---

    def _notify_fit_start(self, X):
        """Notify all callbacks that fitting has started."""
        for cb in self._callbacks:
            cb.on_fit_start(self, X)

    def _notify_iteration(self, info):
        """Notify all callbacks of an iteration completion."""
        for cb in self._callbacks:
            cb.on_iteration_end(self, info)

    def _notify_fit_end(self, info):
        """Notify all callbacks that fitting has ended."""
        for cb in self._callbacks:
            cb.on_fit_end(self, info)


class ScanFitMixin:
    """Mixin providing dual-path training (JIT scan + Python loop).

    Requires the host class to implement:
    - _iteration_step(state, X) -> (new_state, metrics_dict)
    - _build_info(state, iteration) -> dict
    """

    @partial(jit, static_argnums=(0,))
    def _fit_with_scan(self, X, initial_state):
        """JIT-compiled training loop using jax.lax.scan."""
        def scan_fn(state, _):
            return self._iteration_step(state, X)

        final_state, history = jax.lax.scan(
            scan_fn, initial_state, None, length=self.max_iter
        )
        return final_state, history

    def _fit_with_callbacks(self, X, initial_state):
        """Python training loop with callback/patience/restore_best support."""
        self._notify_fit_start(X)
        state = initial_state
        objectives = []
        best_state = None
        best_obj = float('inf')

        for i in range(self.max_iter):
            self._notify_iteration(self._build_info(state, i))
            state, metrics = self._iteration_step(state, X)
            if i == 0:
                if not isinstance(metrics, dict):
                    raise TypeError(
                        f"{type(self).__name__}._iteration_step must return "
                        f"(state, dict), got (state, {type(metrics).__name__})"
                    )
                if 'objective' not in metrics:
                    raise ValueError(
                        f"{type(self).__name__}._iteration_step metrics "
                        f"must contain 'objective'"
                    )
            obj = float(metrics['objective'])
            objectives.append(obj)
            if self.restore_best and obj < best_obj:
                best_obj = obj
                best_state = state
            if state.converged:
                break
            if self.patience is not None and self._check_patience(objectives, self.patience):
                break

        if self.restore_best and best_state is not None:
            state = best_state
            self.best_loss_ = best_obj

        self._notify_fit_end(self._build_info(state, state.iteration))

        history = {
            'objective': jnp.array(objectives),
            'centroid_change': jnp.zeros(len(objectives)),
            'converged': jnp.array(
                [False] * (len(objectives) - 1) + [True]
            ) if objectives else jnp.array([]),
        }
        return state, history

    def _run_training(self, X, initial_state):
        """Route to scan or callbacks path."""
        if self._callbacks or self.patience is not None or self.restore_best:
            return self._fit_with_callbacks(X, initial_state)
        return self._fit_with_scan(X, initial_state)

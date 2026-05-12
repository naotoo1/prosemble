"""
JAX implementation of Possibilistic Fuzzy C-Means (PFCM) clustering algorithm.

PFCM combines FCM and PCM by using both fuzzy membership (U) and typicality (T) matrices.
It provides better clustering by simultaneously considering membership and typicality.

Mathematical Background:
-----------------------
PFCM uses both membership and typicality with weighting parameters a and b.

Objective Function:
    J = Σᵢ Σⱼ [a·uᵢⱼᵐ + b·tᵢⱼⁿ] ||xᵢ - vⱼ||²

Centroid Update:
    vⱼ = Σᵢ[a·uᵢⱼᵐ + b·tᵢⱼⁿ]xᵢ / Σᵢ[a·uᵢⱼᵐ + b·tᵢⱼⁿ]

Membership Update (like FCM):
    uᵢⱼ = 1 / Σₖ(dᵢⱼ/dᵢₖ)^(2/(m-1))

Typicality Update (like PCM):
    tᵢⱼ = 1 / (1 + (b·||xᵢ-vⱼ||²/γⱼ)^(1/(η-1)))

Gamma:
    γⱼ = k·Σᵢ(uᵢⱼᵐ·||xᵢ-vⱼ||²) / Σᵢuᵢⱼᵐ

Author: Prosemble Contributors
License: MIT
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from typing import NamedTuple, Self
import chex

from .fcm import FCM
from .base import FuzzyClusteringBase, ScanFitMixin


class PFCMState(NamedTuple):
    """
    Immutable state for PFCM algorithm.

    Attributes:
        centroids: (c, d) cluster centroids
        U: (n, c) fuzzy membership matrix
        T: (n, c) typicality matrix
        gamma: (c,) scale parameters
        objective: Scalar objective value
        iteration: Current iteration
        converged: Boolean convergence flag
    """
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class PFCM(ScanFitMixin, FuzzyClusteringBase):
    """
    JAX implementation of Possibilistic Fuzzy C-Means clustering.

    PFCM combines fuzzy membership and typicality for robust clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters

    fuzzifier : float, default=2.0
        Fuzzification parameter for membership (m)

    eta : float, default=2.0
        Fuzzification parameter for typicality (η)

    a : float, default=1.0
        Weight for fuzzy membership term

    b : float, default=1.0
        Weight for typicality term

    k : float, default=1.0
        Parameter for gamma computation

    max_iter : int, default=100
        Maximum iterations

    epsilon : float, default=1e-5
        Convergence tolerance

    init_method : str, default='fcm'
        Initialization: 'fcm' or 'random'

    random_seed : int, default=42
        Random seed

    plot_steps : bool, default=False
        Enable live visualization

    show_confidence : bool, default=True
        Show confidence in visualization

    show_pca_variance : bool, default=True
        Show PCA variance in visualization

    save_plot_path : str, default=None
        Path to save final plot

    Attributes
    ----------
    centroids_ : array
        Final cluster centroids

    U_ : array
        Final fuzzy membership matrix

    T_ : array
        Final typicality matrix

    gamma_ : array
        Final scale parameters

    objective_ : float
        Final objective value

    n_iter_ : int
        Number of iterations performed

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('fuzzifier', 'a', 'b', 'eta', 'k', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        eta: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        k: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'fcm',
        random_seed: int = 42,
        distance_fn=None,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str = None,
        **kwargs
    ):
        # Model-specific validation first
        if fuzzifier <= 1.0:
            raise ValueError(f"fuzzifier must be > 1, got {fuzzifier}")
        if eta <= 1.0:
            raise ValueError(f"eta must be > 1, got {eta}")
        if a < 0 or b < 0:
            raise ValueError(f"a and b must be >= 0, got a={a}, b={b}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if init_method not in ['fcm', 'random']:
            raise ValueError(f"init_method must be 'fcm' or 'random', got {init_method}")

        super().__init__(
            n_clusters=n_clusters, max_iter=max_iter, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn, plot_steps=plot_steps,
            show_confidence=show_confidence, show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path, **kwargs
        )

        self.fuzzifier = fuzzifier
        self.eta = eta
        self.a = a
        self.b = b
        self.k = k
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.T_ = None
        self.gamma_ = None
        self.history_ = None

    @partial(jit, static_argnums=(0,))
    def _initialize_matrices(self, X: chex.Array, key: chex.PRNGKey) -> tuple[chex.Array, chex.Array]:
        """Initialize U and T matrices."""
        n_samples = X.shape[0]

        # Initialize both U and T using Dirichlet
        key1, key2 = jax.random.split(key)
        U = jax.random.dirichlet(key1, alpha=jnp.ones(self.n_clusters), shape=(n_samples,))
        T = jax.random.dirichlet(key2, alpha=jnp.ones(self.n_clusters), shape=(n_samples,))

        return U, T

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array, T: chex.Array) -> chex.Array:
        """
        Compute centroids using weighted combination of U and T.

        vⱼ = Σᵢ[a·uᵢⱼᵐ + b·tᵢⱼⁿ]xᵢ / Σᵢ[a·uᵢⱼᵐ + b·tᵢⱼⁿ]
        """
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n, c)
        T_fuzz = jnp.power(T, self.eta)  # (n, c)

        # Weighted combination
        weights = self.a * U_fuzz + self.b * T_fuzz  # (n, c)

        numerator = weights.T @ X  # (c, d)
        denominator = jnp.sum(weights, axis=0, keepdims=True).T  # (c, 1)
        denominator = jnp.maximum(denominator, 1e-10)

        centroids = numerator / denominator
        return centroids

    @partial(jit, static_argnums=(0,))
    def _update_U(self, X: chex.Array, centroids: chex.Array) -> chex.Array:
        """
        Update fuzzy membership matrix (same as FCM).

        uᵢⱼ = 1 / Σₖ(dᵢⱼ/dᵢₖ)^(2/(m-1))
        """
        D = self.distance_fn(X, centroids)
        D = jnp.maximum(D, 1e-10)

        power = 1.0 / (self.fuzzifier - 1)
        ratios = jnp.power(D[:, :, None] / D[:, None, :], power)
        denominators = jnp.sum(ratios, axis=2)
        U = 1.0 / denominators
        U = U / jnp.sum(U, axis=1, keepdims=True)

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_gamma(self, X: chex.Array, U: chex.Array, centroids: chex.Array) -> chex.Array:
        """
        Compute gamma parameters using fuzzy membership.

        γⱼ = k·Σᵢ(uᵢⱼᵐ·||xᵢ-vⱼ||²) / Σᵢuᵢⱼᵐ
        """
        D_sq = self.distance_fn(X, centroids)  # (n, c)
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n, c)

        numerator = jnp.sum(U_fuzz * D_sq, axis=0)  # (c,)
        denominator = jnp.sum(U_fuzz, axis=0)  # (c,)
        denominator = jnp.maximum(denominator, 1e-10)

        gamma = self.k * numerator / denominator
        gamma = jnp.maximum(gamma, 1e-10)

        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_T(self, X: chex.Array, centroids: chex.Array, gamma: chex.Array) -> chex.Array:
        """
        Update typicality matrix.

        tᵢⱼ = 1 / (1 + (b·||xᵢ-vⱼ||²/γⱼ)^(1/(η-1)))
        """
        D_sq = self.distance_fn(X, centroids)  # (n, c)

        power = 1.0 / (self.eta - 1)
        ratio = self.b * D_sq / gamma[None, :]  # (n, c)
        ratio = jnp.maximum(ratio, 0)

        T = 1.0 / (1.0 + jnp.power(ratio, power))

        return T

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self,
        X: chex.Array,
        centroids: chex.Array,
        U: chex.Array,
        T: chex.Array
    ) -> chex.Array:
        """
        Compute PFCM objective function.

        J = Σᵢ Σⱼ [a·uᵢⱼᵐ + b·tᵢⱼⁿ] ||xᵢ - vⱼ||²
        """
        D_sq = self.distance_fn(X, centroids)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.eta)

        weights = self.a * U_fuzz + self.b * T_fuzz
        objective = jnp.sum(weights * D_sq)

        return objective

    @partial(jit, static_argnums=(0,))
    def _check_convergence(
        self,
        centroids_old: chex.Array,
        centroids_new: chex.Array
    ) -> chex.Array:
        """Check convergence based on centroid change."""
        diff = jnp.linalg.norm(centroids_new - centroids_old, ord='fro')
        return diff < self.epsilon

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: PFCMState, X: chex.Array) -> tuple[PFCMState, dict]:
        """Single iteration of PFCM."""
        # Update U (membership)
        U_new = self._update_U(X, state.centroids)

        # Compute gamma
        gamma_new = self._compute_gamma(X, U_new, state.centroids)

        # Update T (typicality)
        T_new = self._update_T(X, state.centroids, gamma_new)

        # Compute new centroids
        centroids_new = self._compute_centroids(X, U_new, T_new)

        # Compute objective
        obj_new = self._compute_objective(X, centroids_new, U_new, T_new)

        # Check convergence
        converged = self._check_convergence(state.centroids, centroids_new)

        new_state = PFCMState(
            centroids=centroids_new,
            U=U_new,
            T=T_new,
            gamma=gamma_new,
            objective=obj_new,
            iteration=state.iteration + 1,
            converged=converged
        )

        metrics = {
            'objective': obj_new,
            'converged': converged
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = (jnp.max(state.U, axis=1) + jnp.max(state.T, axis=1)) / 2.0
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def _initialize_from_fcm(self, X: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Initialize using FCM results."""
        fcm = FCM(
            n_clusters=self.n_clusters,
            fuzzifier=self.fuzzifier,
            max_iter=min(50, self.max_iter),
            epsilon=self.epsilon,
            random_seed=42,
            distance_fn=self.distance_fn,
        )
        fcm.fit(X)

        centroids_init = fcm.centroids_
        U_init = fcm.U_

        # Initialize T similar to U
        T_init = U_init.copy()

        return centroids_init, U_init, T_init

    def fit(self, X: jnp.ndarray, initial_centroids=None, resume=False) -> Self:
        """Fit PFCM model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        initial_centroids : array-like, shape (n_clusters, n_features), optional
            Pre-computed centroids for warm starting
        resume : bool, default=False
            If True, resume from the model's current fitted state
        """
        X = self._validate_input(X)

        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        if not jnp.all(jnp.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")

        # Initialize
        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            U_init = self.U_
            T_init = self.T_
            gamma_init = self.gamma_
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_U(X, centroids_init)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
            T_init = self._update_T(X, centroids_init, gamma_init)
        else:
            if self.init_method == 'fcm':
                centroids_init, U_init, T_init = self._initialize_from_fcm(X)
            else:
                self.key, subkey = jax.random.split(self.key)
                U_init, T_init = self._initialize_matrices(X, subkey)
                centroids_init = self._compute_centroids(X, U_init, T_init)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
        obj_init = self._compute_objective(X, centroids_init, U_init, T_init)

        initial_state = PFCMState(
            centroids=centroids_init,
            U=U_init,
            T=T_init,
            gamma=gamma_init,
            objective=obj_init,
            iteration=0,
            converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.T_ = final_state.T
        self.gamma_ = final_state.gamma
        self.objective_ = final_state.objective
        self.n_iter_ = int(final_state.iteration)

        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict cluster labels."""
        self._check_fitted()

        X = jnp.asarray(X)
        U = self._update_U(X, self.centroids_)
        labels = jnp.argmax(U, axis=1)

        return labels

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict fuzzy membership."""
        self._check_fitted()

        X = jnp.asarray(X)
        return self._update_U(X, self.centroids_)

    def predict_typicality(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict typicality values."""
        self._check_fitted()
        self._check_fitted('gamma_')

        X = jnp.asarray(X)
        return self._update_T(X, self.centroids_, self.gamma_)

    def get_objective_history(self) -> jnp.ndarray:
        """Return objective function history."""
        if self.history_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        return self.history_['objective']

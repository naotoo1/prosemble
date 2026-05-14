"""
JAX-based Adaptive Fuzzy C-Means (AFCM) clustering implementation.

This module provides a GPU-accelerated implementation of AFCM using JAX
with JIT compilation for high performance.
"""

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import chex
from jax import jit, lax

from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin
from prosemble.models.fcm import FCM


class AFCMState(NamedTuple):
    """Immutable state for AFCM iteration.

    Attributes:
        centroids: Cluster centroids, shape (n_clusters, n_features)
        U: Fuzzy membership matrix, shape (n_samples, n_clusters)
        T: Typicality matrix, shape (n_samples, n_clusters)
        gamma: Scale parameters, shape (n_clusters,)
        objective: Current objective function value
        iteration: Current iteration number
        converged: Whether algorithm has converged
    """
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class AFCM(ScanFitMixin, FuzzyClusteringBase):
    """
    Adaptive Fuzzy C-Means clustering with JAX.

    AFCM is an adaptive variant that combines fuzzy and possibilistic approaches
    with specific parameter combinations.

    Key features:
    - Centroids use a·U^m + b·T (T to power 1, not m!)
    - Gamma computed with Euclidean distance (not squared)
    - Exponential T update with parameter b
    - Standard FCM U update

    Algorithm:

    1. Initialize U using FCM
    2. Compute gamma parameters using Euclidean distance
    3. Update T using exponential update
    4. Update U using standard FCM rule
    5. Update centroids using combined fuzzy-possibilistic weights
    6. Repeat until convergence

    Objective function::

        J = Σ_i Σ_j [d²_ij · (a·u_ij^m + b·t_ij)] +
            Σ_j[γ_j · Σ_i(t_ij·log(t_ij) - t_ij)]

    Parameters
    ----------
    n_clusters : int
        Number of clusters (must be >= 2)
    fuzzifier : float, default=2.0
        Fuzziness parameter (must be > 1.0)
    a : float, default=1.0
        Weight for fuzzy membership term (must be > 0)
    b : float, default=1.0
        Weight for typicality term (must be > 0)
    k : float, default=1.0
        Scaling parameter for gamma (must be > 0)
    max_iter : int, default=100
        Maximum number of iterations
    epsilon : float, default=1e-5
        Convergence threshold
    init_method : {'fcm'}, default='fcm'
        Initialization method
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

    Attributes
    ----------
    centroids_ : array, shape (n_clusters, n_features)
        Final cluster centroids
    U_ : array, shape (n_samples, n_clusters)
        Final fuzzy membership matrix
    T_ : array, shape (n_samples, n_clusters)
        Final typicality matrix
    gamma_ : array, shape (n_clusters,)
        Final scale parameters
    n_iter_ : int
        Number of iterations until convergence
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective values at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import AFCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = AFCM(n_clusters=2, fuzzifier=2.0, a=1.0, b=1.0, k=1.0, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('fuzzifier', 'a', 'b', 'k', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        k: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'fcm',
        random_seed: int = 42,
        distance_fn=None,
        patience: int | None = None,
        restore_best: bool = False,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str | None = None,
        callbacks=None,
    ):
        # Validate model-specific parameters
        if fuzzifier <= 1.0:
            raise ValueError("fuzzifier must be > 1.0")
        if a <= 0:
            raise ValueError("a must be > 0")
        if b <= 0:
            raise ValueError("b must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")
        if init_method != 'fcm':
            raise ValueError("init_method must be 'fcm' for AFCM")

        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            epsilon=epsilon,
            random_seed=random_seed,
            distance_fn=distance_fn,
            patience=patience,
            restore_best=restore_best,
            plot_steps=plot_steps,
            show_confidence=show_confidence,
            show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path,
            callbacks=callbacks,
        )

        self.fuzzifier = fuzzifier
        self.a = a
        self.b = b
        self.k = k
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.T_ = None
        self.gamma_ = None

    def _initialize(self, X: chex.Array):
        """Initialize using FCM."""
        n_samples = X.shape[0]

        # Initialize using FCM
        fcm = FCM(
            n_clusters=self.n_clusters,
            fuzzifier=self.fuzzifier,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            random_seed=self.random_seed,
            distance_fn=self.distance_fn,
            plot_steps=False
        )
        fcm.fit(X)

        U = fcm.U_
        centroids = fcm.centroids_

        # Initialize T as zeros
        T = jnp.zeros((n_samples, self.n_clusters))

        return U, T, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_gamma(
        self, X: chex.Array, U: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute gamma using Euclidean distance (not squared!).

        γ_j = k·Σ_i(u_ij^m · d_ij) / Σ_i(u_ij^m)
        """
        D_sq = self.distance_fn(X, centroids)
        D = jnp.sqrt(jnp.maximum(D_sq, 1e-10))  # Euclidean distance

        U_fuzz = jnp.power(U, self.fuzzifier)

        numerator = jnp.sum(U_fuzz * D, axis=0)
        denominator = jnp.sum(U_fuzz, axis=0)

        gamma = self.k * numerator / denominator

        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_T(
        self, X: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Update typicality matrix with exponential and parameter b.

        t_ij = exp(-b·d²_ij/γ_j)
        """
        D_sq = self.distance_fn(X, centroids)
        D_sq = jnp.maximum(D_sq, 1e-10)

        # Exponential update with b parameter
        ratio = self.b * D_sq / gamma[None, :]
        T = jnp.exp(-ratio)

        return T

    @partial(jit, static_argnums=(0,))
    def _update_U(
        self, X: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Update fuzzy membership matrix (standard FCM)."""
        # Compute distances
        D = self.distance_fn(X, centroids)
        D = jnp.maximum(D, 1e-10)

        # Compute power for FCM update
        power = 1.0 / (self.fuzzifier - 1.0)

        # Compute distance ratios
        def compute_membership_row(distances_i):
            ratios = distances_i[:, None] / distances_i[None, :]
            powered_ratios = jnp.power(ratios, power)
            denominators = jnp.sum(powered_ratios, axis=1)
            memberships = 1.0 / denominators
            return memberships

        U = jax.vmap(compute_membership_row)(D)

        # Normalize
        U = U / jnp.sum(U, axis=1, keepdims=True)

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, U: chex.Array, T: chex.Array
    ) -> chex.Array:
        """Compute cluster centroids.

        v_j = Σ_i[a·u_ij^m + b·t_ij]x_i / Σ_i[a·u_ij^m + b·t_ij]

        Note: T is NOT raised to a power!
        """
        U_fuzz = jnp.power(U, self.fuzzifier)

        # Combined weights: a·U^m + b·T
        weights = self.a * U_fuzz + self.b * T

        # Compute centroids
        numerator = weights.T @ X
        denominator = jnp.sum(weights, axis=0, keepdims=True).T

        denominator = jnp.maximum(denominator, 1e-10)
        centroids = numerator / denominator

        return centroids

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, U: chex.Array, T: chex.Array,
        centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Compute AFCM objective function.

        J = Σ_i Σ_j [d²_ij · (a·u_ij^m + b·t_ij)] +
            Σ_j[γ_j · Σ_i(t_ij·log(t_ij) - t_ij)]
        """
        D_sq = self.distance_fn(X, centroids)
        U_fuzz = jnp.power(U, self.fuzzifier)

        # First term: Σ_i Σ_j [d²_ij · (a·u_ij^m + b·t_ij)]
        weights = self.a * U_fuzz + self.b * T
        term1 = jnp.sum(D_sq * weights)

        # Second term: Σ_j[γ_j · Σ_i(t·log(t) - t)]
        T_safe = jnp.maximum(T, 1e-10)
        entropy_like = T * jnp.log(T_safe) - T
        inner_sum = jnp.sum(entropy_like, axis=0)
        term2 = jnp.sum(gamma * inner_sum)

        objective = term1 + term2

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(
        self, state: AFCMState, X: chex.Array
    ) -> tuple[AFCMState, dict]:
        """Single AFCM iteration step."""
        # Update T
        T_new = self._update_T(X, state.centroids, state.gamma)

        # Update U
        U_new = self._update_U(X, state.centroids)

        # Update centroids
        centroids_new = self._compute_centroids(X, U_new, T_new)

        # Recompute gamma with new U and centroids
        gamma_new = self._compute_gamma(X, U_new, centroids_new)

        # Compute objective
        objective = self._compute_objective(X, U_new, T_new, centroids_new, gamma_new)

        # Check convergence
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = AFCMState(
            centroids=centroids_new,
            U=U_new,
            T=T_new,
            gamma=gamma_new,
            objective=objective,
            iteration=state.iteration + 1,
            converged=converged
        )

        metrics = {
            'objective': objective,
            'centroid_change': centroid_change,
            'converged': converged
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = jnp.max(state.U * state.T, axis=1)
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit AFCM model to data."""
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)

        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            U_init = self._update_U(X, centroids_init)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
            T_init = self._update_T(X, centroids_init, gamma_init)
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_U(X, centroids_init)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
            T_init = self._update_T(X, centroids_init, gamma_init)
        else:
            U_init, T_init, centroids_init = self._initialize(X)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)

        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init, gamma_init)
        initial_state = AFCMState(
            centroids=centroids_init, U=U_init, T=T_init, gamma=gamma_init,
            objective=initial_objective, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.T_ = final_state.T
        self.gamma_ = final_state.gamma
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels for new data."""
        self._check_fitted()

        U = self._update_U(X, self.centroids_)
        labels = jnp.argmax(U, axis=1)
        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict fuzzy membership probabilities."""
        self._check_fitted()

        X = jnp.asarray(X)

        U = self._update_U(X, self.centroids_)
        return U

    def get_typicality(self, X: chex.Array) -> chex.Array:
        """Compute typicality values."""
        self._check_fitted()

        X = jnp.asarray(X)

        T = self._update_T(X, self.centroids_, self.gamma_)
        return T

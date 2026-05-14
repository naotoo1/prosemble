"""
JAX-based Improved Possibilistic C-Means 2 (IPCM2) clustering implementation.

This module provides a GPU-accelerated implementation of IPCM2 using JAX
with JIT compilation for high performance.
"""

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import chex
from jax import jit, lax

from prosemble.models.fcm import FCM
from prosemble.models.base import FuzzyClusteringBase


class IPCM2State(NamedTuple):
    """Immutable state for IPCM2 iteration.

    Attributes:
        centroids: Cluster centroids, shape (n_clusters, n_features)
        U: Fuzzy membership matrix, shape (n_samples, n_clusters)
        T: Typicality matrix, shape (n_samples, n_clusters)
        gamma: Scale parameters, shape (n_clusters,)
        objective: Current objective function value
        iteration: Current iteration number
        converged: Whether algorithm has converged
        phase: Current phase (0 or 1)
    """
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool
    phase: int


class IPCM2(FuzzyClusteringBase):
    """
    Improved Possibilistic C-Means 2 clustering with JAX.

    IPCM2 is a variant of IPCM with key differences:
    - Uses exponential T update: :math:`t_{ij} = \\exp(-d_{ij}^2 / \\gamma_j)`
    - Centroids use :math:`U^{m_f} \\cdot T` (T without power!)
    - Modified U update with exponential distance
    - Different objective function

    Algorithm (Phase 0):

    1. Initialize U using FCM, T = 0
    2. Compute gamma parameters from fuzzy membership
    3. Update T using exponential update
    4. Update U with modified distance
    5. Update centroids using combined U and T weights
    6. Repeat until convergence

    Algorithm (Phase 1):

    7. Recompute gamma using both U and T
    8. Continue iterations with new gamma

    Objective function:

    .. math::

        J = \\sum_i \\sum_j u_{ij}^{m_f} \\cdot t_{ij} \\cdot d_{ij}^2 + \\sum_j \\gamma_j \\sum_i (t_{ij} \\log t_{ij} - t_{ij} + 1) \\cdot u_{ij}^{m_f}

    Parameters
    ----------
    fuzzifier : float, default=2.0
        Fuzziness parameter for U matrix (m_f, must be > 1.0).
    tipifier : float, default=2.0
        Possibilistic parameter for T matrix (m_p, must be > 1.0).
    init_method : {'fcm'}, default='fcm'
        Method for initializing U matrix.
    n_clusters : int
        Number of clusters (must be >= 2).
    max_iter : int
        Maximum number of iterations.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed for reproducibility.
    distance_fn : callable, optional
        Pairwise distance function. Default: squared Euclidean.
    patience : int, optional
        Epochs with no improvement before early stopping. Default: None.
    restore_best : bool
        If True, restore centroids from the lowest-objective epoch.
        Default: False.
    plot_steps : bool
        Whether to visualize clustering progress. Default: False.
    show_confidence : bool
        Whether to show confidence in visualization. Default: True.
    show_pca_variance : bool
        Whether to show PCA variance in visualization. Default: True.
    save_plot_path : str, optional
        Path to save final plot.
    callbacks : list, optional
        List of Callback objects for monitoring/visualization.

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
        Total number of iterations
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective values at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import IPCM2
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = IPCM2(n_clusters=2, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)
    """

    _hyperparams = ('fuzzifier', 'tipifier', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        tipifier: float = 2.0,
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
        if tipifier <= 1.0:
            raise ValueError("tipifier must be > 1.0")
        if init_method != 'fcm':
            raise ValueError("init_method must be 'fcm' for IPCM2")

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
        self.tipifier = tipifier
        self.init_method = init_method

        # Fitted attributes
        self.U_ = None
        self.T_ = None
        self.gamma_ = None

    def _initialize_phase0(self, X: chex.Array):
        """Initialize for phase 0 using FCM."""
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
    def _compute_gamma_phase0(
        self, X: chex.Array, U: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute gamma for phase 0.

        .. math::

            \\gamma_j = \\frac{\\sum_i u_{ij}^{m_f} \\cdot d_{ij}^2}{\\sum_i u_{ij}^{m_f}}
        """
        D_sq = self.distance_fn(X, centroids)
        U_fuzz = jnp.power(U, self.fuzzifier)

        numerator = jnp.sum(U_fuzz * D_sq, axis=0)
        denominator = jnp.maximum(jnp.sum(U_fuzz, axis=0), 1e-10)

        gamma = numerator / denominator

        return gamma

    @partial(jit, static_argnums=(0,))
    def _compute_gamma_phase1(
        self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute gamma for phase 1.

        .. math::

            \\gamma_j = \\frac{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}^{m_p} \\cdot d_{ij}^2}{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}^{m_p}}
        """
        D_sq = self.distance_fn(X, centroids)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.tipifier)

        prod = U_fuzz * T_fuzz

        numerator = jnp.sum(prod * D_sq, axis=0)
        denominator = jnp.maximum(jnp.sum(prod, axis=0), 1e-10)

        gamma = numerator / denominator

        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_T(
        self, X: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Update typicality matrix with exponential.

        .. math::

            t_{ij} = \\exp\\left(-\\frac{d_{ij}^2}{\\gamma_j}\\right)
        """
        D_sq = self.distance_fn(X, centroids)
        D_sq = jnp.maximum(D_sq, 1e-10)

        # Exponential update
        ratio = D_sq / gamma[None, :]
        T = jnp.exp(-ratio)

        return T

    @partial(jit, static_argnums=(0,))
    def _update_U(
        self, X: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Update fuzzy membership matrix (IPCM2-specific).

        .. math::

            u_{ij} = \\frac{\\left(\\frac{1}{\\gamma_j (1 - \\exp(-d_{ij}^2/\\gamma_j))}\\right)^{2/(m_f-1)}}{\\sum_k \\left(\\frac{1}{\\gamma_k (1 - \\exp(-d_{ik}^2/\\gamma_k))}\\right)^{2/(m_f-1)}}
        """
        D_sq = self.distance_fn(X, centroids)
        D_sq = jnp.maximum(D_sq, 1e-10)

        # Compute modified distance: gamma_j*(1-exp(-d^2_ij/gamma_j))
        ratio = D_sq / gamma[None, :]
        exp_term = jnp.exp(-ratio)
        modified_dist = gamma[None, :] * (1.0 - exp_term)
        modified_dist = jnp.maximum(modified_dist, 1e-10)

        # Compute base values: 1/modified_dist
        base_values = 1.0 / modified_dist

        # Compute power
        power = 2.0 / (self.fuzzifier - 1.0)

        # Raise to power
        powered_values = jnp.power(base_values, power)

        # Normalize
        denominators = jnp.sum(powered_values, axis=1, keepdims=True)
        U = powered_values / denominators

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, U: chex.Array, T: chex.Array
    ) -> chex.Array:
        """Compute cluster centroids.

        .. math::

            v_j = \\frac{\\sum_i u_{ij}^{m_f} \\cdot t_{ij} \\cdot x_i}{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}}

        Note: T is NOT raised to power m_p here!
        """
        U_fuzz = jnp.power(U, self.fuzzifier)

        # Product of U^m_f and T (NOT T^m_p!)
        weights = U_fuzz * T

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
        """Compute IPCM2 objective function.

        .. math::

            J = \\sum_i \\sum_j u_{ij}^{m_f} \\cdot t_{ij} \\cdot d_{ij}^2 + \\sum_j \\gamma_j \\sum_i (t_{ij} \\log t_{ij} - t_{ij} + 1) \\cdot u_{ij}^{m_f}
        """
        D_sq = self.distance_fn(X, centroids)
        U_fuzz = jnp.power(U, self.fuzzifier)

        # First term: sum_i sum_j [u_ij^m_f * t_ij * d^2_ij]
        term1 = jnp.sum(U_fuzz * T * D_sq)

        # Second term: sum_j[gamma_j * sum_i((t*log(t) - t + 1) * u^m_f)]
        # Handle log(0) by clamping T
        T_safe = jnp.maximum(T, 1e-10)
        entropy_like = T * jnp.log(T_safe) - T + 1.0
        inner_sum = jnp.sum(entropy_like * U_fuzz, axis=0)
        term2 = jnp.sum(gamma * inner_sum)

        objective = term1 + term2

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: IPCM2State, X: chex.Array) -> IPCM2State:
        """Single IPCM2 iteration step."""
        # Update T
        T_new = self._update_T(X, state.centroids, state.gamma)

        # Update U
        U_new = self._update_U(X, state.centroids, state.gamma)

        # Update centroids
        centroids_new = self._compute_centroids(X, U_new, T_new)

        # Compute objective
        objective = self._compute_objective(X, U_new, T_new, centroids_new, state.gamma)

        # Check convergence
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = IPCM2State(
            centroids=centroids_new,
            U=U_new,
            T=T_new,
            gamma=state.gamma,
            objective=objective,
            iteration=state.iteration + 1,
            converged=converged,
            phase=state.phase
        )

        return new_state

    def _run_phase(
        self, X: chex.Array, U_init: chex.Array, T_init: chex.Array,
        centroids_init: chex.Array, gamma_init: chex.Array, phase: int
    ):
        """Run one phase of IPCM2."""
        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init, gamma_init)

        state = IPCM2State(
            centroids=centroids_init,
            U=U_init,
            T=T_init,
            gamma=gamma_init,
            objective=initial_objective,
            iteration=0,
            converged=False,
            phase=phase
        )

        states_history = [state]
        objectives = [float(state.objective)]
        best_state = None
        best_obj = float('inf')

        for i in range(self.max_iter):
            self._notify_iteration(self._build_info(state, state.iteration))
            state = self._iteration_step(state, X)
            states_history.append(state)
            obj = float(state.objective)
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

        return state, states_history

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = jnp.max(state.U * state.T, axis=1)
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit IPCM2 model to data."""
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)
        self._notify_fit_start(X)

        if resume:
            # Skip phase 0, run only phase 1 with fitted state
            self._check_fitted()
            gamma_1 = self._compute_gamma_phase1(
                X, self.U_, self.T_, self.centroids_
            )
            state_final, history_phase1 = self._run_phase(
                X, self.U_, self.T_, self.centroids_, gamma_1, phase=1
            )
            self._notify_fit_end(self._build_info(state_final, state_final.iteration))
            all_objectives = [s.objective for s in history_phase1]
            self.objective_history_ = jnp.array(all_objectives)
        else:
            if initial_centroids is not None:
                centroids_init = self._validate_initial_centroids(X, initial_centroids)
                # Derive U and T from centroids
                gamma_0 = self._compute_gamma_phase0(X,
                    jnp.ones((X.shape[0], self.n_clusters)) / self.n_clusters,
                    centroids_init)
                T_init = self._update_T(X, centroids_init, gamma_0)
                U_init = self._update_U(X, centroids_init, gamma_0)
            else:
                U_init, T_init, centroids_init = self._initialize_phase0(X)

            # Phase 0
            gamma_0 = self._compute_gamma_phase0(X, U_init, centroids_init)
            state_phase0, history_phase0 = self._run_phase(
                X, U_init, T_init, centroids_init, gamma_0, phase=0
            )

            # Phase 1
            gamma_1 = self._compute_gamma_phase1(
                X, state_phase0.U, state_phase0.T, state_phase0.centroids
            )
            state_final, history_phase1 = self._run_phase(
                X, state_phase0.U, state_phase0.T,
                state_phase0.centroids, gamma_1, phase=1
            )

            self._notify_fit_end(self._build_info(state_final, state_final.iteration))
            all_objectives = [s.objective for s in history_phase0] + [s.objective for s in history_phase1]
            self.objective_history_ = jnp.array(all_objectives)

        self.centroids_ = state_final.centroids
        self.U_ = state_final.U
        self.T_ = state_final.T
        self.gamma_ = state_final.gamma
        self.n_iter_ = int(state_final.iteration)
        self.objective_ = float(state_final.objective)

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels for new data."""
        self._check_fitted()

        T = self._update_T(X, self.centroids_, self.gamma_)
        U = self._update_U(X, self.centroids_, self.gamma_)

        labels = jnp.argmax(U, axis=1)
        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict fuzzy membership probabilities."""
        self._check_fitted()

        X = jnp.asarray(X)

        U = self._update_U(X, self.centroids_, self.gamma_)
        return U

    def get_typicality(self, X: chex.Array) -> chex.Array:
        """Compute typicality values."""
        self._check_fitted()

        X = jnp.asarray(X)

        T = self._update_T(X, self.centroids_, self.gamma_)
        return T

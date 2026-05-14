"""
JAX-based Improved Possibilistic C-Means (IPCM) clustering implementation.

This module provides a GPU-accelerated implementation of IPCM using JAX
with JIT compilation for high performance.
"""

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import chex
from jax import jit, lax

from prosemble.models.base import FuzzyClusteringBase
from prosemble.models.fcm import FCM


class IPCMState(NamedTuple):
    """Immutable state for IPCM iteration.

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


class IPCM(FuzzyClusteringBase):
    """
    Improved Possibilistic C-Means clustering with JAX.

    IPCM uses a two-phase approach to improve clustering performance:
    - Phase 0: Initialize :math:`\\gamma` using fuzzy membership only
    - Phase 1: Refine :math:`\\gamma` using both membership and typicality

    Key differences from PCM:
    - Uses product of :math:`U^{m_f}` and :math:`T^{m_p}` in centroid computation
    - Modified :math:`U` update that depends on :math:`T`
    - Two-phase :math:`\\gamma` computation

    Algorithm (Phase 0):

    1. Initialize :math:`U` using FCM, :math:`T = 0`
    2. Compute :math:`\\gamma` parameters from fuzzy membership
    3. Update typicality matrix :math:`T`
    4. Update membership matrix :math:`U`
    5. Update centroids using combined U and T weights
    6. Repeat until convergence

    Algorithm (Phase 1):

    7. Recompute :math:`\\gamma` using both :math:`U` and :math:`T`
    8. Continue iterations with new gamma

    Objective function:

    .. math::

        J = \\sum_i \\sum_j u_{ij}^{m_f} \\cdot t_{ij}^{m_p} \\cdot d_{ij}^2 + \\sum_j \\gamma_j \\sum_i (1 - t_{ij})^{m_p} \\cdot u_{ij}^{m_f}

    Parameters
    ----------
    fuzzifier : float, default=2.0
        Fuzziness parameter for :math:`U` matrix (:math:`m_f`, must be > 1.0).
    tipifier : float, default=2.0
        Possibilistic parameter for :math:`T` matrix (:math:`m_p`, must be > 1.0).
    k : float, default=1.0
        Scaling parameter for :math:`\\gamma` in phase 1 (must be > 0).
    init_method : {'fcm'}, default='fcm'
        Method for initializing :math:`U` matrix (must use FCM).
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
        Total number of iterations across both phases
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective value at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import IPCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = IPCM(n_clusters=2, fuzzifier=2.0, tipifier=2.0, k=1.0, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)
    """

    _hyperparams = ('fuzzifier', 'tipifier', 'k', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        tipifier: float = 2.0,
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
        if tipifier <= 1.0:
            raise ValueError("tipifier must be > 1.0")
        if k <= 0:
            raise ValueError("k must be > 0")
        if init_method != 'fcm':
            raise ValueError("init_method must be 'fcm' for IPCM")

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
        self.k = k
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.T_ = None
        self.gamma_ = None

    def _initialize_phase0(self, X: chex.Array):
        """Initialize for phase 0 using FCM.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            Tuple of (U, T, centroids)
        """
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
        """Compute :math:`\\gamma` for phase 0.

        .. math::

            \\gamma_j = \\frac{\\sum_i u_{ij}^{m_f} \\cdot d_{ij}^2}{\\sum_i u_{ij}^{m_f}}

        Args:
            X: Input data, shape (n_samples, n_features)
            U: Fuzzy membership matrix, shape (n_samples, n_clusters)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            gamma: shape (n_clusters,)
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)

        # Fuzzify U
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n_samples, n_clusters)

        # Compute gamma for each cluster
        numerator = jnp.sum(U_fuzz * D_sq, axis=0)  # (n_clusters,)
        denominator = jnp.maximum(jnp.sum(U_fuzz, axis=0), 1e-10)  # (n_clusters,)

        gamma = numerator / denominator

        return gamma

    @partial(jit, static_argnums=(0,))
    def _compute_gamma_phase1(
        self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute :math:`\\gamma` for phase 1.

        .. math::

            \\gamma_j = k \\cdot \\frac{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}^{m_p} \\cdot d_{ij}^2}{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}^{m_p}}

        Args:
            X: Input data, shape (n_samples, n_features)
            U: Fuzzy membership matrix, shape (n_samples, n_clusters)
            T: Typicality matrix, shape (n_samples, n_clusters)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            gamma: shape (n_clusters,)
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)

        # Fuzzify U and T
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.tipifier)

        # Product of memberships
        prod = U_fuzz * T_fuzz  # (n_samples, n_clusters)

        # Compute gamma for each cluster
        numerator = jnp.sum(prod * D_sq, axis=0)  # (n_clusters,)
        denominator = jnp.maximum(jnp.sum(prod, axis=0), 1e-10)  # (n_clusters,)

        gamma = self.k * numerator / denominator

        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_T(
        self, X: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Update typicality matrix.

        .. math::

            t_{ij} = \\frac{1}{1 + \\left(\\frac{d_{ij}^2}{\\gamma_j}\\right)^{1/(m_p-1)}}

        Args:
            X: Input data, shape (n_samples, n_features)
            centroids: Current centroids, shape (n_clusters, n_features)
            gamma: Scale parameters, shape (n_clusters,)

        Returns:
            T: Updated typicality matrix, shape (n_samples, n_clusters)
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)
        D_sq = jnp.maximum(D_sq, 1e-10)  # Avoid division by zero

        # Compute power
        power = 1.0 / (self.tipifier - 1.0)

        # Compute ratio and typicality
        ratio = D_sq / gamma[None, :]  # (n_samples, n_clusters)
        T = 1.0 / (1.0 + jnp.power(ratio, power))

        return T

    @partial(jit, static_argnums=(0,))
    def _update_U(
        self, X: chex.Array, T: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Update fuzzy membership matrix (IPCM-specific).

        .. math::

            u_{ij} = \\frac{\\left(\\frac{1}{d_{ij}^2} \\cdot t_{ij}^{m_p-1}\\right)^{1/(m_f-1)}}{\\sum_k \\left(\\frac{1}{d_{ik}^2} \\cdot t_{ik}^{m_p-1}\\right)^{1/(m_f-1)}}

        Args:
            X: Input data, shape (n_samples, n_features)
            T: Typicality matrix, shape (n_samples, n_clusters)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            U: Updated membership matrix, shape (n_samples, n_clusters)
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)
        D_sq = jnp.maximum(D_sq, 1e-10)  # Avoid division by zero

        # Compute T^(m_p-1)
        T_pow = jnp.power(T, self.tipifier - 1.0)  # (n_samples, n_clusters)

        # Compute base values: (1/d^2_ij * t_ij^(m_p-1))
        base_values = (1.0 / D_sq) * T_pow  # (n_samples, n_clusters)

        # Compute power
        power = 1.0 / (self.fuzzifier - 1.0)

        # Raise to power
        powered_values = jnp.power(base_values, power)  # (n_samples, n_clusters)

        # Normalize
        denominators = jnp.sum(powered_values, axis=1, keepdims=True)  # (n_samples, 1)
        U = powered_values / denominators

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, U: chex.Array, T: chex.Array
    ) -> chex.Array:
        """Compute cluster centroids.

        .. math::

            v_j = \\frac{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}^{m_p} \\cdot x_i}{\\sum_i u_{ij}^{m_f} \\cdot t_{ij}^{m_p}}

        Args:
            X: Input data, shape (n_samples, n_features)
            U: Fuzzy membership matrix, shape (n_samples, n_clusters)
            T: Typicality matrix, shape (n_samples, n_clusters)

        Returns:
            centroids: shape (n_clusters, n_features)
        """
        # Fuzzify U and T
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.tipifier)

        # Product of memberships
        weights = U_fuzz * T_fuzz  # (n_samples, n_clusters)

        # Compute centroids
        numerator = weights.T @ X  # (n_clusters, n_features)
        denominator = jnp.sum(weights, axis=0, keepdims=True).T  # (n_clusters, 1)

        # Handle empty clusters
        denominator = jnp.maximum(denominator, 1e-10)
        centroids = numerator / denominator

        return centroids

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, U: chex.Array, T: chex.Array,
        centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Compute IPCM objective function.

        .. math::

            J = \\sum_i \\sum_j u_{ij}^{m_f} \\cdot t_{ij}^{m_p} \\cdot d_{ij}^2 + \\sum_j \\gamma_j \\sum_i (1 - t_{ij})^{m_p} \\cdot u_{ij}^{m_f}

        Args:
            X: Input data
            U: Fuzzy membership matrix
            T: Typicality matrix
            centroids: Current centroids
            gamma: Scale parameters

        Returns:
            objective: Scalar objective value
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)

        # Fuzzify U and T
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.tipifier)

        # First term: sum_i sum_j [u_ij^m_f * t_ij^m_p * d^2_ij]
        term1 = jnp.sum(U_fuzz * T_fuzz * D_sq)

        # Second term: sum_j[gamma_j * sum_i((1-t_ij)^m_p * u_ij^m_f)]
        one_minus_T = 1.0 - T
        one_minus_T_fuzz = jnp.power(one_minus_T, self.tipifier)
        inner_sum = jnp.sum(one_minus_T_fuzz * U_fuzz, axis=0)  # (n_clusters,)
        term2 = jnp.sum(gamma * inner_sum)

        objective = term1 + term2

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: IPCMState, X: chex.Array) -> IPCMState:
        """Single IPCM iteration step.

        Args:
            state: Current IPCM state
            X: Input data

        Returns:
            new_state: Updated IPCM state
        """
        # Update T
        T_new = self._update_T(X, state.centroids, state.gamma)

        # Update U
        U_new = self._update_U(X, T_new, state.centroids)

        # Update centroids
        centroids_new = self._compute_centroids(X, U_new, T_new)

        # Compute objective
        objective = self._compute_objective(X, U_new, T_new, centroids_new, state.gamma)

        # Check convergence
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = IPCMState(
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
    ) -> IPCMState:
        """Run one phase of IPCM.

        Args:
            X: Input data
            U_init: Initial U matrix
            T_init: Initial T matrix
            centroids_init: Initial centroids
            gamma_init: Initial gamma
            phase: Phase number (0 or 1)

        Returns:
            final_state: Final state after phase convergence
        """
        # Initial objective
        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init, gamma_init)

        # Initial state
        state = IPCMState(
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
        """Fit IPCM model to data."""
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
                # Derive U from centroids, T starts as zeros
                T_init = jnp.zeros((X.shape[0], self.n_clusters))
                gamma_0 = self._compute_gamma_phase0(X,
                    jnp.ones((X.shape[0], self.n_clusters)) / self.n_clusters,
                    centroids_init)
                T_init = self._update_T(X, centroids_init, gamma_0)
                U_init = self._update_U(X, T_init, centroids_init)
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

        # Store results
        self.centroids_ = state_final.centroids
        self.U_ = state_final.U
        self.T_ = state_final.T
        self.gamma_ = state_final.gamma
        self.n_iter_ = int(state_final.iteration)
        self.objective_ = float(state_final.objective)

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels for new data.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            labels: Cluster labels, shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted
        """
        self._check_fitted()

        # Compute T for new data
        T = self._update_T(X, self.centroids_, self.gamma_)

        # Compute U for new data
        U = self._update_U(X, T, self.centroids_)

        # Assign to cluster with highest membership
        labels = jnp.argmax(U, axis=1)

        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict fuzzy membership probabilities (U matrix).

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            U: Fuzzy membership matrix, shape (n_samples, n_clusters)

        Raises:
            ValueError: If model has not been fitted
        """
        self._check_fitted()

        X = jnp.asarray(X)

        # Compute T for new data
        T = self._update_T(X, self.centroids_, self.gamma_)

        # Compute U for new data
        U = self._update_U(X, T, self.centroids_)

        return U

    def get_typicality(self, X: chex.Array) -> chex.Array:
        """Compute typicality values (T matrix).

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            T: Typicality matrix, shape (n_samples, n_clusters)

        Raises:
            ValueError: If model has not been fitted
        """
        self._check_fitted()

        X = jnp.asarray(X)

        T = self._update_T(X, self.centroids_, self.gamma_)

        return T

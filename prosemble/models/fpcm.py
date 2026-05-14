"""
JAX-based Fuzzy Possibilistic C-Means (FPCM) clustering implementation.

This module provides a GPU-accelerated implementation of FPCM using JAX
with JIT compilation for high performance.
"""

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import chex
from jax import jit

from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin
from prosemble.models.fcm import FCM


class FPCMState(NamedTuple):
    """Immutable state for FPCM iteration.

    Attributes:
        centroids: Cluster centroids, shape (n_clusters, n_features)
        U: Fuzzy membership matrix, shape (n_samples, n_clusters)
        T: Possibilistic typicality matrix, shape (n_samples, n_clusters)
        objective: Current objective function value
        iteration: Current iteration number
        converged: Whether algorithm has converged
    """
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class FPCM(ScanFitMixin, FuzzyClusteringBase):
    """
    Fuzzy Possibilistic C-Means clustering with JAX.

    FPCM maintains TWO matrices: U (fuzzy membership) and T (typicality).
    U has row-sum-to-1 constraint (standard FCM), while T has column-sum-to-1
    constraint per the original Pal, Pal & Bezdek (1997) formulation.

    Algorithm:

    1. Initialize U and T (randomly or using FCM)
    2. Update centroids using combined fuzzy and typicality weights
    3. Update U using FCM rule with fuzzifier m (row-normalized)
    4. Update T with column-normalization
    5. Repeat until convergence

    Objective function:

    .. math::

        J = \\sum_i \\sum_j \\left[u_{ij}^m + t_{ij}^\\eta\\right] \\|x_i - v_j\\|^2

    Reference:
        Pal, N. R., Pal, K., & Bezdek, J. C. (1997).
        A mixed c-means clustering model. FUZZ-IEEE.

    Parameters
    ----------
    fuzzifier : float, default=2.0
        Fuzziness parameter for U matrix (must be > 1.0).
    eta : float, default=2.0
        Fuzziness parameter for T matrix (must be > 1.0).
    init_method : {'random', 'fcm'}, default='fcm'
        Method for initializing U and T matrices.
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
        Final possibilistic typicality matrix
    n_iter_ : int
        Number of iterations until convergence
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective value at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import FPCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = FPCM(n_clusters=2, fuzzifier=2.0, eta=2.0, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)
    >>> U = model.predict_proba(X)
    >>> T = model.get_typicality(X)
    """

    _hyperparams = ('fuzzifier', 'eta', 'init_method')
    _fitted_array_names = ('U_', 'T_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        eta: float = 2.0,
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
        if eta <= 1.0:
            raise ValueError("eta must be > 1.0")
        if init_method not in ['random', 'fcm']:
            raise ValueError("init_method must be 'random' or 'fcm'")

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
        self.eta = eta
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.T_ = None

    def _initialize_matrices(self, X: chex.Array):
        """Initialize U and T matrices.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            Tuple of (U, T, centroids)
        """
        n_samples = X.shape[0]

        if self.init_method == 'random':
            # Random initialization (Dirichlet distribution ensures row sums = 1)
            alpha = jnp.ones(self.n_clusters)
            U = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))
            T = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))

            # Compute initial centroids
            centroids = self._compute_centroids(X, U, T)

        elif self.init_method == 'fcm':
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

            # Initialize T randomly
            alpha = jnp.ones(self.n_clusters)
            T = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        return U, T, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, U: chex.Array, T: chex.Array
    ) -> chex.Array:
        """Compute cluster centroids.

        .. math::

            v_j = \\frac{\\sum_i \\left[u_{ij}^m + t_{ij}^\\eta\\right] x_i}{\\sum_i \\left[u_{ij}^m + t_{ij}^\\eta\\right]}

        Args:
            X: Input data, shape (n_samples, n_features)
            U: Fuzzy membership matrix, shape (n_samples, n_clusters)
            T: Typicality matrix, shape (n_samples, n_clusters)

        Returns:
            centroids: shape (n_clusters, n_features)
        """
        # Fuzzify U and T
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n_samples, n_clusters)
        T_fuzz = jnp.power(T, self.eta)  # (n_samples, n_clusters)

        # Combined weights
        weights = U_fuzz + T_fuzz  # (n_samples, n_clusters)

        # Compute centroids
        numerator = weights.T @ X  # (n_clusters, n_features)
        denominator = jnp.sum(weights, axis=0, keepdims=True).T  # (n_clusters, 1)

        centroids = numerator / denominator

        return centroids

    @partial(jit, static_argnums=(0,))
    def _update_fuzzy_matrix(
        self, X: chex.Array, centroids: chex.Array, fuzzifier: float
    ) -> chex.Array:
        """Update fuzzy membership matrix using FCM rule.

        Standard FCM update:

        .. math::

            u_{ij} = \\frac{1}{\\sum_k \\left(\\frac{d_{ij}}{d_{ik}}\\right)^{2/(m-1)}}

        Args:
            X: Input data, shape (n_samples, n_features)
            centroids: Current centroids, shape (n_clusters, n_features)
            fuzzifier: Fuzziness parameter

        Returns:
            U: Updated membership matrix, shape (n_samples, n_clusters)
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)

        # Add small epsilon to avoid division by zero
        D_sq = jnp.maximum(D_sq, 1e-10)

        # Compute power for FCM update
        power = 2.0 / (fuzzifier - 1.0)

        # Compute distance ratios: (d_ij / d_ik)^power for all k
        # For each i,j: sum over k of (D[i,j] / D[i,k])^power
        def compute_membership_row(distances_i):
            # distances_i: (n_clusters,)
            # For each j, compute: 1 / sum_k (d_ij / d_ik)^power
            ratios = distances_i[:, None] / distances_i[None, :]  # (n_clusters, n_clusters)
            powered_ratios = jnp.power(ratios, power)  # (n_clusters, n_clusters)
            denominators = jnp.sum(powered_ratios, axis=1)  # (n_clusters,)
            memberships = 1.0 / denominators
            return memberships

        U = jax.vmap(compute_membership_row)(D_sq)  # (n_samples, n_clusters)

        # Normalize to ensure row sums = 1 (for numerical stability)
        U = U / jnp.sum(U, axis=1, keepdims=True)

        return U

    @partial(jit, static_argnums=(0,))
    def _update_typicality_matrix(
        self, X: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Update typicality matrix with column-sum-to-1 constraint.

        Per Pal, Pal & Bezdek (1997):

        .. math::

            t_{ij} = \\frac{(1/d_{ij}^2)^{1/(\\eta-1)}}{\\sum_i (1/d_{ij}^2)^{1/(\\eta-1)}}

        Each column j sums to 1 across samples (:math:`\\sum_i t_{ij} = 1`).

        Args:
            X: Input data, shape (n_samples, n_features)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            T: Typicality matrix, shape (n_samples, n_clusters)
        """
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)
        D_sq = jnp.maximum(D_sq, 1e-10)

        power = 1.0 / (self.eta - 1.0)
        inv_dist_powered = jnp.power(1.0 / D_sq, power)  # (n_samples, n_clusters)

        # Normalize over samples (axis=0) so each column sums to 1
        col_sums = jnp.maximum(jnp.sum(inv_dist_powered, axis=0, keepdims=True), 1e-10)
        T = inv_dist_powered / col_sums

        return T

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute FPCM objective function.

        .. math::

            J = \\sum_i \\sum_j \\left[u_{ij}^m + t_{ij}^\\eta\\right] \\|x_i - v_j\\|^2

        Args:
            X: Input data, shape (n_samples, n_features)
            U: Fuzzy membership matrix, shape (n_samples, n_clusters)
            T: Typicality matrix, shape (n_samples, n_clusters)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            objective: Scalar objective value
        """
        # Compute squared distances
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)

        # Fuzzify U and T
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.eta)

        # Combined weights
        weights = U_fuzz + T_fuzz

        # Weighted distances
        weighted_distances = weights * D_sq

        # Sum over all elements
        objective = jnp.sum(weighted_distances)

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: FPCMState, X: chex.Array) -> tuple[FPCMState, dict]:
        """Single FPCM iteration step.

        Args:
            state: Current FPCM state
            X: Input data, shape (n_samples, n_features)

        Returns:
            new_state: Updated FPCM state
            metrics: Dictionary of iteration metrics
        """
        # Update U with fuzzifier m (row-normalized)
        U_new = self._update_fuzzy_matrix(X, state.centroids, self.fuzzifier)

        # Update T with column-normalization (Pal et al. 1997)
        T_new = self._update_typicality_matrix(X, state.centroids)

        # Update centroids
        centroids_new = self._compute_centroids(X, U_new, T_new)

        # Compute objective
        objective = self._compute_objective(X, U_new, T_new, centroids_new)

        # Check convergence based on centroid change
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = FPCMState(
            centroids=centroids_new,
            U=U_new,
            T=T_new,
            objective=objective,
            iteration=state.iteration + 1,
            converged=converged
        )

        metrics = {
            'objective': new_state.objective,
            'centroid_change': centroid_change,
            'converged': new_state.converged,
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

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit FPCM model to data.

        Args:
            X: Input data, shape (n_samples, n_features)
            initial_centroids: Optional initial centroids for warm starting
            resume: If True, resume from fitted state

        Returns:
            self: Fitted model

        Raises:
            ValueError: If n_samples < n_clusters
        """
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)

        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            U_init = self._update_fuzzy_matrix(X, centroids_init, self.fuzzifier)
            T_init = self._update_typicality_matrix(X, centroids_init)
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_fuzzy_matrix(X, centroids_init, self.fuzzifier)
            T_init = self._update_typicality_matrix(X, centroids_init)
        else:
            U_init, T_init, centroids_init = self._initialize_matrices(X)

        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init)
        initial_state = FPCMState(
            centroids=centroids_init, U=U_init, T=T_init,
            objective=initial_objective, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.T_ = final_state.T
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']

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

        # Compute U for new data
        U = self._update_fuzzy_matrix(X, self.centroids_, self.fuzzifier)

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

        U = self._update_fuzzy_matrix(X, self.centroids_, self.fuzzifier)

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

        T = self._update_typicality_matrix(X, self.centroids_)

        return T

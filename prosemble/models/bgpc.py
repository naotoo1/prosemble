"""
JAX implementation of Bayesian Graded Possibilistic C-Means (BGPC)

This is a GPU-accelerated implementation using JAX.
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from functools import partial
from typing import NamedTuple, Self

import chex
import jax
import jax.numpy as jnp
from jax import jit
from jax import lax

from prosemble.core.distance import batch_squared_euclidean, batch_euclidean


class BGPCState(NamedTuple):
    """State for BGPC optimization loop"""
    centroids: chex.Array
    U: chex.Array
    V: chex.Array
    iteration: int
    converged: bool
    alpha: float
    beta: float


class BGPC:
    """
    Bayesian Graded Possibilistic C-Means (BGPC) with JAX

    BGPC uses exponential weighting with time-decaying alpha and beta parameters.

    Algorithm:

    1. Compute membership weights using exponential distance
    2. Normalize memberships using partition function Z
    3. Update centroids as weighted mean of data
    4. Update beta and alpha with decay schedules
    5. Repeat until convergence

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
    alpha_init : float, default=1.0
        Initial alpha parameter
    beta_init : float, default=0.1
        Initial beta parameter (starting value for decay)
    beta_final : float, default=10.0
        Final beta parameter (ending value for decay)
    init : str, default='fcm'
        Initialization method: 'random', 'fcm', or 'kmeans++'
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        alpha_init: float = 1.0,
        beta_init: float = 0.1,
        beta_final: float = 10.0,
        init: str = 'fcm',
        random_state: int | None = None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.init = init
        self.random_state = random_state

        # Fitted attributes
        self.centroids_ = None
        self.U_ = None
        self.V_ = None
        self.n_iter_ = 0
        self.alpha_ = None
        self.beta_ = None

    @partial(jit, static_argnums=(0,))
    def _compute_beta_decay(self, iteration: int) -> float:
        """
        Compute beta decay: :math:`\\beta(t) = 0.1 \\cdot (\\beta_f / 0.1)^{t/T}`.

        :math:`\\beta` starts at 0.1 and grows to :math:`\\beta_f` over iterations.
        """
        ratio = iteration / self.max_iter
        beta = self.beta_init * jnp.power(self.beta_final / self.beta_init, ratio)
        return beta

    @partial(jit, static_argnums=(0,))
    def _compute_alpha_decay(self, iteration: int) -> float:
        """
        Compute alpha decay: :math:`\\alpha(t) = (1 - \\beta_f)(1 + \\exp(t - T) + \\alpha_0)`.

        :math:`\\alpha` decays over iterations.
        """
        alpha = (1 - self.beta_final) * (1 + jnp.exp(iteration - self.max_iter) + self.alpha_init)
        return alpha

    @partial(jit, static_argnums=(0,))
    def _compute_V_matrix(self, X: chex.Array, centroids: chex.Array, beta: float) -> chex.Array:
        """
        Compute V matrix: :math:`V_{ij} = \\exp(-d(x_i, v_j) / \\beta)`.

        Uses Euclidean distance (not squared).
        """
        D_sq = batch_squared_euclidean(X, centroids)
        D = jnp.sqrt(jnp.maximum(D_sq, 1e-10))
        V = jnp.exp(-D / beta)
        return V

    @partial(jit, static_argnums=(0,))
    def _compute_z_value(self, v_i: chex.Array, alpha: float) -> float:
        """
        Compute :math:`Z_i` for a single data point based on :math:`V_i` values.

        Logic from original:

        - If :math:`\\sum_k v_{ik}^{1/\\alpha} > 1`: :math:`z_i = (\\sum_k v_{ik}^{1/\\alpha})^\\alpha`
        - If :math:`\\sum_k v_{ik}^\\alpha < 1`: :math:`z_i = (\\sum_k v_{ik}^\\alpha)^{1/\\alpha}`
        - Otherwise: :math:`z_i = 1`
        """
        v_pow_inv_alpha = jnp.power(v_i, 1.0 / alpha)
        v_pow_alpha = jnp.power(v_i, alpha)

        sum_inv = jnp.sum(v_pow_inv_alpha)
        sum_alpha = jnp.sum(v_pow_alpha)

        # Compute z based on conditions
        z = jnp.where(
            sum_inv > 1.0,
            jnp.power(sum_inv, alpha),
            jnp.where(
                sum_alpha < 1.0,
                jnp.power(sum_alpha, 1.0 / alpha),
                1.0
            )
        )
        return z

    @partial(jit, static_argnums=(0,))
    def _compute_Z_list(self, V: chex.Array, alpha: float) -> chex.Array:
        """Compute Z values for all data points"""
        # Vectorized version using vmap
        compute_z_vmap = jax.vmap(lambda v_i: self._compute_z_value(v_i, alpha))
        Z = compute_z_vmap(V)
        return Z

    @partial(jit, static_argnums=(0,))
    def _update_U_matrix(self, V: chex.Array, Z: chex.Array) -> chex.Array:
        """
        Update U matrix: :math:`U_{ij} = V_{ij} / Z_i`.
        """
        U = V / (Z[:, None] + 1e-10)
        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array) -> chex.Array:
        """
        Compute centroids: :math:`v_j = \\sum_i u_{ij} x_i / \\sum_i u_{ij}`.
        """
        numerator = U.T @ X
        denominator = jnp.sum(U, axis=0, keepdims=True).T
        centroids = numerator / jnp.maximum(denominator, 1e-10)
        return centroids

    @partial(jit, static_argnums=(0,))
    def _initialize_centroids_random(self, X: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Random initialization"""
        n_samples = X.shape[0]
        indices = jax.random.choice(key, n_samples, shape=(self.n_clusters,), replace=False)
        return X[indices]

    @partial(jit, static_argnums=(0,))
    def _initialize_centroids_kmeanspp(self, X: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """K-means++ initialization"""
        n_samples = X.shape[0]

        # First centroid: random
        key, subkey = jax.random.split(key)
        first_idx = jax.random.choice(subkey, n_samples)
        centroids = X[first_idx:first_idx+1]

        # Remaining centroids
        def body_fn(i, state):
            cents, k = state
            # Compute distances to nearest centroid
            D_sq = batch_squared_euclidean(X, cents)
            min_distances = jnp.min(D_sq, axis=1)

            # Sample proportional to squared distance
            k, subk = jax.random.split(k)
            probs = min_distances / jnp.sum(min_distances)
            next_idx = jax.random.choice(subk, n_samples, p=probs)
            new_cent = X[next_idx:next_idx+1]
            cents = jnp.concatenate([cents, new_cent], axis=0)
            return cents, k

        centroids, _ = lax.fori_loop(0, self.n_clusters - 1, body_fn, (centroids, key))
        return centroids

    def _initialize_centroids_fcm(self, X: chex.Array) -> chex.Array:
        """Initialize using FCM (requires importing FCM)"""
        from .fcm import FCM

        random_seed = self.random_state if self.random_state is not None else 42
        fcm = FCM(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_seed=random_seed
        )
        fcm.fit(X)
        return fcm.centroids_

    def _initialize_centroids(self, X: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Initialize centroids based on init method"""
        if self.init == 'random':
            return self._initialize_centroids_random(X, key)
        elif self.init == 'kmeans++':
            return self._initialize_centroids_kmeanspp(X, key)
        elif self.init == 'fcm':
            return self._initialize_centroids_fcm(X)
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    @partial(jit, static_argnums=(0,))
    def _check_convergence(self, centroids_old: chex.Array, centroids_new: chex.Array) -> bool:
        """Check if centroids have converged"""
        diff = jnp.linalg.norm(centroids_new - centroids_old)
        return diff <= self.tol

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: BGPCState, X: chex.Array) -> BGPCState:
        """Single iteration of BGPC"""
        # Compute beta and alpha for this iteration
        beta = self._compute_beta_decay(state.iteration)
        alpha = self._compute_alpha_decay(state.iteration)

        # Compute V matrix
        V = self._compute_V_matrix(X, state.centroids, beta)

        # Compute Z values
        Z = self._compute_Z_list(V, alpha)

        # Update U matrix
        U = self._update_U_matrix(V, Z)

        # Update centroids
        centroids_new = self._compute_centroids(X, U)

        # Check convergence
        converged = self._check_convergence(state.centroids, centroids_new)

        return BGPCState(
            centroids=centroids_new,
            U=U,
            V=V,
            iteration=state.iteration + 1,
            converged=converged,
            alpha=alpha,
            beta=beta
        )

    @partial(jit, static_argnums=(0,))
    def _optimize(self, X: chex.Array, initial_centroids: chex.Array) -> BGPCState:
        """Run BGPC optimization loop"""
        # Initialize state
        n_samples = X.shape[0]
        initial_V = jnp.zeros((n_samples, self.n_clusters))
        initial_U = jnp.ones((n_samples, self.n_clusters)) / self.n_clusters

        initial_state = BGPCState(
            centroids=initial_centroids,
            U=initial_U,
            V=initial_V,
            iteration=0,
            converged=False,
            alpha=self.alpha_init,
            beta=self.beta_init
        )

        # Optimization loop
        def cond_fn(state):
            return jnp.logical_and(
                state.iteration < self.max_iter,
                jnp.logical_not(state.converged)
            )

        def body_fn(state):
            return self._iteration_step(state, X)

        final_state = lax.while_loop(cond_fn, body_fn, initial_state)
        return final_state

    def fit(self, X: chex.Array) -> Self:
        """
        Fit BGPC model to data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self
        """
        X = jnp.asarray(X)

        # Initialize random key
        if self.random_state is not None:
            key = jax.random.PRNGKey(self.random_state)
        else:
            key = jax.random.PRNGKey(0)

        # Initialize centroids
        initial_centroids = self._initialize_centroids(X, key)

        # Run optimization
        final_state = self._optimize(X, initial_centroids)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.V_ = final_state.V
        self.n_iter_ = int(final_state.iteration)
        self.alpha_ = float(final_state.alpha)
        self.beta_ = float(final_state.beta)

        return self

    @partial(jit, static_argnums=(0,))
    def _predict_labels(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels (hard assignment)"""
        D_sq = batch_squared_euclidean(X, self.centroids_)
        labels = jnp.argmin(D_sq, axis=1)
        return labels

    def predict(self, X: chex.Array) -> chex.Array:
        """
        Predict cluster labels for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict

        Returns
        -------
        labels : array of shape (n_samples,)
            Cluster labels
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._predict_labels(X)

    @partial(jit, static_argnums=(0,))
    def _predict_proba(self, X: chex.Array) -> chex.Array:
        """Compute U matrix (membership probabilities) for new data"""
        # Compute V matrix
        V = self._compute_V_matrix(X, self.centroids_, self.beta_)

        # Compute Z values
        Z = self._compute_Z_list(V, self.alpha_)

        # Compute U matrix
        U = self._update_U_matrix(V, Z)
        return U

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """
        Predict membership probabilities for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict

        Returns
        -------
        U : array of shape (n_samples, n_clusters)
            Membership probabilities
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._predict_proba(X)

    @partial(jit, static_argnums=(0,))
    def _get_typicality(self, X: chex.Array) -> chex.Array:
        """Compute V matrix (typicality values) for new data"""
        V = self._compute_V_matrix(X, self.centroids_, self.beta_)
        return V

    def get_typicality(self, X: chex.Array) -> chex.Array:
        """
        Get typicality values for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute typicality for

        Returns
        -------
        V : array of shape (n_samples, n_clusters)
            Typicality values
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._get_typicality(X)

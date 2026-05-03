"""
Fuzzy C-Means clustering example using Iris Data with JAX.

This example demonstrates:
- Loading datasets with JAX-native DATA_JAX
- JAX-based train/test splitting
- FCM_JAX clustering
- Pure JAX implementation (no NumPy/sklearn)
"""

import jax
import jax.numpy as jnp
from prosemble.datasets import load_breast_cancer_jax
from prosemble.models.jax import FCM_JAX


def train_test_split_jax(X, y, test_size=0.2, random_seed=42):
    """JAX-native train/test split."""
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Shuffle indices
    key = jax.random.PRNGKey(random_seed)
    indices = jax.random.permutation(key, n_samples)

    # Split
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


# Load data (JAX arrays directly)
dataset = load_breast_cancer_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup FCM model
fcm = FCM_JAX(
    n_clusters=2,
    fuzzifier=2.0,
    max_iter=100,
    epsilon=1e-5,
    random_seed=42,
    plot_steps=True
)

# Fit model
print("\nTraining FCM...")
fcm.fit(X_train)

# Results
print(f"\nObjective History (last 5): {fcm.get_objective_history()[-5:]}")
print(f"Converged in {fcm.n_iter_} iterations")
print(f"Final objective: {fcm.objective_:.4f}")

# Predictions
train_labels = fcm.predict(X_train)
test_labels = fcm.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Centroids
centroids = fcm.final_centroids()
print(f"\nCentroids shape: {centroids.shape}")
print(f"Centroids:\n{centroids}")

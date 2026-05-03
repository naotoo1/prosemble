"""
Hard C-Means clustering example using Iris Data with JAX.

This example demonstrates HCM with pure JAX implementation.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import HCM

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup HCM model
hcm = HCM(
    n_clusters=3,
    max_iter=100,
    epsilon=1e-5,
    random_seed=42,
    plot_steps=True
)

# Fit model
print("\nTraining HCM...")
hcm.fit(X_train)

# Results
print(f"\nObjective History (last 5): {hcm.get_objective_history()[-5:]}")
print(f"Converged in {hcm.n_iter_} iterations")
print(f"Final objective: {hcm.objective_:.4f}")

# Predictions
train_labels = hcm.predict(X_train)
test_labels = hcm.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Centroids
centroids = hcm.final_centroids()
print(f"\nCentroids shape: {centroids.shape}")

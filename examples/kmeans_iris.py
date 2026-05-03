"""
K-means++ clustering example using Iris Data with JAX.

This example demonstrates KMeansPlusPlus (K-means++ initialization)
with pure JAX implementation.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import KMeansPlusPlus

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup K-means++ model
kmeans = KMeansPlusPlus(
    n_clusters=3,
    max_iter=100,
    epsilon=1e-5,
    random_seed=42,
    plot_steps=True
)

# Fit model
print("\nTraining K-means++...")
kmeans.fit(X_train)

# Results
print(f"\nObjective History (last 5): {kmeans.get_objective_history()[-5:]}")
print(f"Converged in {kmeans.n_iter_} iterations")
print(f"Final objective: {kmeans.objective_:.4f}")

# Predictions
train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Centroids
centroids = kmeans.final_centroids()
print(f"\nCentroids shape: {centroids.shape}")
print("\nNote: K-means++ provides better initialization than random selection,")
print("often leading to faster convergence and better final clustering.")

"""
Kernel Improved Possibilistic C-Means clustering example using Iris Data with JAX.

This example demonstrates KIPCM with pure JAX implementation.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import KIPCM

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup KIPCM model
kipcm = KIPCM(
    n_clusters=3,
    fuzzifier=2.0,
    tipifier=2.0,
    k=1.0,
    sigma=1.0,
    max_iter=100,
    epsilon=1e-5,
    init_method='fcm',
    random_seed=42,
    plot_steps=True
)

# Fit model
print("\nTraining KIPCM...")
kipcm.fit(X_train)

# Results
print(f"\nObjective History (last 5): {kipcm.get_objective_history()[-5:]}")
print(f"Converged in {kipcm.n_iter_} iterations")
print(f"Final objective: {kipcm.objective_:.4f}")

# Predictions
train_labels = kipcm.predict(X_train)
test_labels = kipcm.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Centroids
centroids = kipcm.final_centroids()
print(f"\nCentroids shape: {centroids.shape}")

"""
Improved Possibilistic C-Means 2 clustering example using Iris Data with JAX.

This example demonstrates IPCM2_JAX with pure JAX implementation.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils_jax import train_test_split_jax
from prosemble.models.jax import IPCM2_JAX

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup IPCM2 model
ipcm2 = IPCM2_JAX(
    n_clusters=3,
    fuzzifier=2.0,
    tipifier=2.0,
    max_iter=100,
    epsilon=1e-5,
    init_method='fcm',
    random_seed=42,
    plot_steps=True
)

# Fit model
print("\nTraining IPCM2...")
ipcm2.fit(X_train)

# Results
print(f"\nObjective History (last 5): {ipcm2.get_objective_history()[-5:]}")
print(f"Converged in {ipcm2.n_iter_} iterations")
print(f"Final objective: {ipcm2.objective_:.4f}")

# Predictions
train_labels = ipcm2.predict(X_train)
test_labels = ipcm2.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Centroids
centroids = ipcm2.final_centroids()
print(f"\nCentroids shape: {centroids.shape}")

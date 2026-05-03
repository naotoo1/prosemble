"""
Self-Organizing Maps (SOM) example using Iris Data with JAX.

This example demonstrates SOM_JAX with pure JAX implementation.
SOM creates a low-dimensional representation of high-dimensional data.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils_jax import train_test_split_jax
from prosemble.models.jax import SOM_JAX

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup SOM model
som = SOM_JAX(
    grid_size=10,
    max_iter=1000,
    learning_rate=0.5,
    sigma=1.0,
    random_state=42
)

# Fit model
print("\nTraining SOM...")
som.fit(X_train)

# Results
print(f"\nGrid size: {som.grid_size}x{som.grid_size}")
print(f"Trained for {som.n_iter_} iterations")

# Transform to grid coordinates
train_coords = som.transform(X_train)
test_coords = som.transform(X_test)

print(f"\nTrain coordinates (BMU): {train_coords.shape}")
print(f"Test coordinates (BMU): {test_coords.shape}")

# Get SOM weights
som_weights = som.som_
print(f"\nSOM weights shape: {som_weights.shape}")

# Example: Show first few BMU coordinates
print(f"\nFirst 5 train BMU coordinates:")
print(train_coords[:5])

print("\nNote: SOM is an unsupervised algorithm that creates a topological")
print("mapping of the input space to a 2D grid, preserving neighborhood relationships.")

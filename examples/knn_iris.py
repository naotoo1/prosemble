"""
K-Nearest Neighbors classification example using Iris Data with JAX.

This example demonstrates KNN_JAX with pure JAX implementation.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils_jax import train_test_split_jax
from prosemble.models.jax import KNN_JAX

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup KNN model
knn = KNN_JAX(n_neighbors=5)

# Fit model
print("\nTraining KNN...")
knn.fit(X_train, y_train)

# Predictions
train_labels = knn.predict(X_train)
test_labels = knn.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Probabilities
train_proba = knn.get_proba(X_train)
test_proba = knn.get_proba(X_test)

print(f"\nTrain probabilities (confidence): {train_proba.shape}")
print(f"Test probabilities (confidence): {test_proba.shape}")

# Full probability distribution
train_proba_full = knn.predict_proba(X_train)
test_proba_full = knn.predict_proba(X_test)

print(f"\nTrain full probabilities: {train_proba_full.shape}")
print(f"Test full probabilities: {test_proba_full.shape}")

# Accuracy calculation
train_accuracy = jnp.mean(train_labels == y_train)
test_accuracy = jnp.mean(test_labels == y_test)

print(f"\nTrain accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

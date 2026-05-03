"""
Bagging Gaussian Prototype Classifier example using Iris Data with JAX.

This example demonstrates BGPC with pure JAX implementation.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax, accuracy_score_jax
from prosemble.models import BGPC

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.3, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup BGPC model
bgpc = BGPC(
    n_clusters=3,
    max_iter=10,
    alpha_init=1.0,
    random_state=42
)

# Fit model (unsupervised)
print("\nTraining BGPC...")
bgpc.fit(X_train)

# Results
print(f"\nTraining iterations: {bgpc.n_iter_}")

# Predictions
train_pred = bgpc.predict(X_train)
test_pred = bgpc.predict(X_test)

# Accuracy (comparing cluster assignments to true labels)
train_acc = accuracy_score_jax(y_train, train_pred)
test_acc = accuracy_score_jax(y_test, test_pred)

print(f"\nTrain accuracy: {train_acc:.2%}")
print(f"Test accuracy: {test_acc:.2%}")
print(f"Centroids shape: {bgpc.centroids_.shape}")

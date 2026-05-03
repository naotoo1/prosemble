"""
Noise Possibilistic C-Means (NPC) classification example using Iris Data with JAX.

This example demonstrates NPC_JAX with pure JAX implementation.
NPC is a supervised prototype-based classifier.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils_jax import train_test_split_jax
from prosemble.models.jax import NPC_JAX

# Load data (JAX arrays directly)
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup NPC model
npc = NPC_JAX(
    n_classes=3,
    max_iter=10,
    tol=0.8,
    random_state=42
)

# Fit model
print("\nTraining NPC...")
npc.fit(X_train, y_train)

# Results
print(f"\nConverged in {npc.n_iter_} iterations")

# Predictions
train_labels = npc.predict(X_train)
test_labels = npc.predict(X_test)

print(f"\nTrain predictions: {train_labels.shape}")
print(f"Test predictions: {test_labels.shape}")

# Probabilities (using softmin)
train_proba = npc.predict_proba(X_train)
test_proba = npc.predict_proba(X_test)

print(f"\nTrain probabilities: {train_proba.shape}")
print(f"Test probabilities: {test_proba.shape}")

# Distance space
train_distances = npc.get_distance_space(X_train)
test_distances = npc.get_distance_space(X_test)

print(f"\nTrain distance space: {train_distances.shape}")
print(f"Test distance space: {test_distances.shape}")

# Accuracy calculation
train_accuracy = jnp.mean(train_labels == y_train)
test_accuracy = jnp.mean(test_labels == y_test)

print(f"\nTrain accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Prototypes
prototypes = npc.prototypes_
print(f"\nPrototypes shape: {prototypes.shape}")
print("\nNote: NPC is a supervised classifier that learns one prototype per class.")

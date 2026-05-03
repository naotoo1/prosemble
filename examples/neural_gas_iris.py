"""
Neural Gas example using Iris Data with JAX.

This example demonstrates:
- Unsupervised topology-preserving clustering with Neural Gas
- Rank-based neighborhood: closer prototypes get larger updates
- Exponential decay for both learning rate and neighborhood range
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import NeuralGas

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup Neural Gas model
model = NeuralGas(
    n_prototypes=10,
    max_iter=100,
    lr_init=0.5,
    lr_final=0.01,
    lambda_init=5.0,
    lambda_final=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining Neural Gas...")
model.fit(X_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")
print(f"Loss decreased: {model.loss_history_[0]:.4f} -> {model.loss_history_[-1]:.4f}")

# Predictions (assign to nearest prototype)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print(f"\nTrain cluster assignments: {train_preds.shape}")
print(f"Unique clusters used: {len(jnp.unique(train_preds))}")

# Distance to all prototypes
distances = model.transform(X_test[:5])
print(f"\nDistance matrix (5 samples x {model.n_prototypes} prototypes):")
print(f"  Shape: {distances.shape}")
print(f"  Min distances: {jnp.min(distances, axis=1)}")

# Prototypes
print(f"\nPrototypes shape: {model.prototypes_.shape}")

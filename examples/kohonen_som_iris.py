"""
Kohonen Self-Organizing Map (SOM) example using Iris Data with JAX.

This example demonstrates:
- Standard Kohonen SOM with Gaussian neighborhood
- 2D grid topology with exponential decay of sigma and learning rate
- BMU (Best Matching Unit) mapping to grid coordinates
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import KohonenSOM

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup Kohonen SOM model (5x5 grid)
model = KohonenSOM(
    grid_height=5,
    grid_width=5,
    sigma_init=3.0,
    sigma_final=0.5,
    lr_init=0.5,
    lr_final=0.01,
    max_iter=100,
    random_seed=42,
)

# Fit model
print("\nTraining Kohonen SOM...")
model.fit(X_train)

# Results
print(f"Grid: {model.grid_height}x{model.grid_width} = {model.grid_height * model.grid_width} nodes")
print(f"Converged in {model.n_iter_} iterations")
print(f"Final quantization error: {model.loss_:.4f}")

# BMU assignments
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
print(f"\nNodes used: {len(jnp.unique(train_preds))} / {model.grid_height * model.grid_width}")

# BMU grid coordinates
train_coords = model.bmu_map(X_train)
print(f"\nBMU coordinates (first 10 train samples):")
for i in range(min(10, len(train_coords))):
    row, col = int(train_coords[i, 0]), int(train_coords[i, 1])
    print(f"  Sample {i}: grid ({row}, {col}), class {int(y_train[i])}")

# Distance to all prototypes
distances = model.transform(X_test[:5])
print(f"\nDistance matrix shape: {distances.shape}")

# Prototypes
print(f"\nPrototype weights shape: {model.prototypes_.shape}")

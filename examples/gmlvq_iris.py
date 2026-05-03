"""
Generalized Matrix LVQ (GMLVQ) example using Iris Data with JAX.

This example demonstrates:
- Metric learning with a global Omega transformation matrix
- The learned Lambda = Omega^T @ Omega reveals feature relevances
- Dimensionality reduction via the latent projection
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import GMLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape} ({X.shape[1]} features)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup GMLVQ model (project to 2D latent space)
model = GMLVQ(
    n_prototypes_per_class=1,
    latent_dim=2,
    max_iter=100,
    lr=0.001,
    random_seed=42,
)

# Fit model
print("\nTraining GMLVQ...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = float(jnp.mean(train_preds == y_train))
test_acc = float(jnp.mean(test_preds == y_test))

print(f"\nTrain accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")

# Learned metric
print(f"\nOmega matrix shape: {model.omega_matrix.shape}")
print(f"Lambda matrix (Omega^T @ Omega):\n{model.lambda_matrix}")

# Feature relevances from diagonal of Lambda
lambda_diag = jnp.diag(model.lambda_matrix)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("\nFeature relevances:")
for name, rel in zip(feature_names, lambda_diag):
    print(f"  {name}: {rel:.4f}")

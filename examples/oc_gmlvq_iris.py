"""
One-Class GMLVQ (OC-GMLVQ) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with global Omega matrix adaptation
- Learns a linear transformation capturing feature correlations
- Distance: d(x, w) = ||Omega(x - w)||^2
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import OCGMLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

target_label = 0
print(f"Dataset: {X.shape}")
print(f"Target class: {target_label}")

# Setup OC-GMLVQ model
model = OCGMLVQ(
    n_prototypes=3,
    target_label=target_label,
    latent_dim=2,
    beta=10.0,
    max_iter=100,
    lr=0.001,
    random_seed=42,
)

# Fit model
print("\nTraining OC-GMLVQ...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
test_preds = model.predict(X_test)
preds_reject = model.predict_with_reject(
    X_test, upper=0.8, lower=0.2, reject_label=-1
)
n_rejected = int(jnp.sum(preds_reject == -1))
print(f"\nRejected: {n_rejected}/{len(X_test)}")

# Learned matrices
print(f"\nOmega matrix shape: {model.omega_matrix.shape}")
print(f"Lambda matrix shape: {model.lambda_matrix.shape}")

# Feature importance from diagonal of Lambda = Omega^T @ Omega
feature_importance = jnp.diag(model.lambda_matrix)
print(f"Feature importance: {feature_importance}")

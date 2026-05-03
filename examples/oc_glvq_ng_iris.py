"""
One-Class GLVQ with Neural Gas (OC-GLVQ-NG) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with Neural Gas neighborhood cooperation
- All prototypes participate in updates weighted by rank
- Gamma decays from wide to narrow cooperation during training
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import OCGLVQ_NG

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

# Setup OC-GLVQ-NG model
model = OCGLVQ_NG(
    n_prototypes=3,
    target_label=target_label,
    beta=10.0,
    gamma_init=3.0,
    gamma_final=0.01,
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining OC-GLVQ-NG...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions with reject option
test_preds = model.predict(X_test)
preds_reject = model.predict_with_reject(
    X_test, upper=0.8, lower=0.2, reject_label=-1
)
n_rejected = int(jnp.sum(preds_reject == -1))
print(f"\nRejected: {n_rejected}/{len(X_test)}")

# Visibility radii
print(f"Visibility radii: {model.visibility_radii}")
print(f"Prototypes shape: {model.prototypes_.shape}")

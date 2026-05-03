"""LVQMLN on Iris dataset.

Demonstrates the LVQ Multi-Layer Network: an MLP backbone learns
a latent space, prototypes live in that space, and GLVQ loss
trains both jointly.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.models import LVQMLN

# Load data
dataset = load_iris_jax()
X = dataset.input_data
y = dataset.labels

# Fit LVQMLN: 4D input -> [10] hidden -> 2D latent
model = LVQMLN(
    hidden_sizes=[10],
    latent_dim=2,
    activation='sigmoid',
    n_prototypes_per_class=2,
    max_iter=200,
    lr=0.01,
    random_seed=42,
)
model.fit(X, y)

print(f"Trained {model.n_iter_} epochs, loss={model.loss_:.4f}")
print(f"Prototypes shape: {model.prototypes_.shape}")

# Predict
preds = model.predict(X)
accuracy = jnp.mean(preds == y)
print(f"Training accuracy: {accuracy:.2%}")

# Transform to latent space
latent = model.transform(X)
print(f"Latent representation shape: {latent.shape}")

# Class probabilities
proba = model.predict_proba(X)
print(f"Probability shape: {proba.shape}")
print(f"Mean max probability: {jnp.max(proba, axis=1).mean():.3f}")

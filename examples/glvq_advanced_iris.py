"""
Advanced GLVQ example: chained optimizer + unequal prototypes on Iris.

This example demonstrates:
- Unequal prototype distribution via dict: {class: count}
- Chained optax transforms: gradient clipping + weight decay + adam
- Learning rate scheduling with cosine decay
- Prototype win ratio analysis
- Class probability inspection
"""

import optax
import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import GLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {jnp.unique(y)}")

# Build a chained optimizer:
#   1. Centralize gradients (subtract mean, helps convergence)
#   2. Clip gradient norm (prevent exploding gradients)
#   3. Adam with cosine-decayed learning rate
schedule = optax.cosine_decay_schedule(init_value=0.01, decay_steps=200)
optimizer = optax.chain(
    optax.centralize(),
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule),
)

# Unequal prototype distribution:
# 3 prototypes for class 0, 2 for class 1, 1 for class 2
model = GLVQ(
    n_prototypes_per_class={0: 3, 1: 2, 2: 1},
    optimizer=optimizer,
    max_iter=200,
    random_seed=42,
)

# Train
print("\nTraining GLVQ with chained optimizer...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Accuracy
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
train_acc = float(jnp.mean(train_preds == y_train))
test_acc = float(jnp.mean(test_preds == y_test))
print(f"\nTrain accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")

# Prototype details
print(f"\nPrototypes shape: {model.prototypes_.shape}")
print(f"Prototype labels: {model.prototype_labels_}")

# Win ratios: how often each prototype wins on correctly classified samples
win_ratios = model.prototype_win_ratios(X_train, y_train)
print(f"\nPrototype win ratios:")
for i, (label, ratio) in enumerate(zip(model.prototype_labels_, win_ratios)):
    print(f"  Prototype {i} (class {int(label)}): {float(ratio):.3f}")

# Class probabilities for first 5 test samples
proba = model.predict_proba(X_test[:5])
print(f"\nClass probabilities (first 5 test samples):")
for i in range(5):
    pred = int(jnp.argmax(proba[i]))
    true = int(y_test[i])
    conf = float(jnp.max(proba[i]))
    print(f"  Sample {i}: true={true}, pred={pred}, conf={conf:.3f}, probs={proba[i]}")

"""
GLVQ with Unequal Prototypes Per Class — Iris Dataset.

This example demonstrates three ways to specify prototype distribution:
- int: same count for all classes (e.g. 2)
- list: per-class counts by index (e.g. [2, 2, 1])
- dict: per-class counts by label (e.g. {0: 2, 1: 2, 2: 1})
"""

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

def run_glvq(distribution, label):
    """Train and evaluate GLVQ with the given prototype distribution."""
    print(f"\n{'='*60}")
    print(f"Distribution format: {label}")
    print(f"Value: {distribution}")
    print('='*60)

    model = GLVQ(
        n_prototypes_per_class=distribution,
        max_iter=100,
        lr=0.01,
        random_seed=42,
    )

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = float(jnp.mean(train_preds == y_train))
    test_acc = float(jnp.mean(test_preds == y_test))

    print(f"Converged in {model.n_iter_} iterations")
    print(f"Final loss: {model.loss_:.4f}")
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}")
    print(f"Prototypes shape: {model.prototypes_.shape}")
    print(f"Prototype labels: {model.prototype_labels_}")
    for c in range(3):
        n = int(jnp.sum(model.prototype_labels_ == c))
        print(f"  Class {c}: {n} prototypes")


# 1. Dict format: {class_label: count}
run_glvq({0: 2, 1: 2, 2: 1}, "dict")

# 2. List format: [count_class_0, count_class_1, count_class_2]
run_glvq([2, 2, 1], "list")

# 3. Int format: same count for all classes
run_glvq(2, "int (uniform)")

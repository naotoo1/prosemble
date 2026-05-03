"""Visualize GLVQ decision boundaries on Iris dataset."""

from prosemble.datasets import load_iris_jax
from prosemble.models import GLVQ
from prosemble.core.vis import plot_decision_boundary_2d, plot_lvq_summary

# Load data
dataset = load_iris_jax()
X = dataset.input_data
y = dataset.labels

# Fit GLVQ
model = GLVQ(
    n_prototypes_per_class=2, max_iter=100, lr=0.01, random_seed=42,
)
model.fit(X, y)
print(f"Trained {model.n_iter_} epochs, loss={model.loss_:.4f}")

# Decision boundary
fig1 = plot_decision_boundary_2d(model, X, y)

# Full summary (boundary + loss curve)
fig2 = plot_lvq_summary(model, X, y)

import matplotlib.pyplot as plt
fig2.savefig('vis_glvq_iris.png', dpi=150, bbox_inches='tight')
print("Saved vis_glvq_iris.png")

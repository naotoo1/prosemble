"""Visualize Fuzzy C-Means clustering on Iris dataset."""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from prosemble.datasets import load_iris_jax
from prosemble.models import FCM

# Load data
dataset = load_iris_jax()
X = dataset.input_data
y = dataset.labels

# Fit FCM
model = FCM(n_clusters=3, m=2.0, max_iter=150, random_seed=42)
model.fit(X)
print(f"Trained {model.n_iter_} iterations, loss={model.loss_:.4f}")

# Get results
U = np.array(model.membership_matrix_)
centers = np.array(model.cluster_centers_)
preds = np.array(model.predict(X))
X_np = np.array(X)
y_np = np.array(y)

# PCA for 2D projection
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_np)
centers_2d = pca.transform(centers)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Cluster assignments vs true labels
ax = axes[0]
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=preds, cmap='Set1', s=30, alpha=0.7)
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='X', s=200,
           edgecolors='white', linewidths=1.5, zorder=5)
ax.set_title('FCM Cluster Assignments')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# 2. Membership heatmap
ax = axes[1]
im = ax.imshow(U.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_title('Membership Matrix')
ax.set_xlabel('Sample index')
ax.set_ylabel('Cluster')
ax.set_yticks(range(U.shape[1]))
fig.colorbar(im, ax=ax, shrink=0.8)

# 3. Loss curve
ax = axes[2]
ax.plot(model.loss_history_, color='steelblue', linewidth=1.5)
ax.set_title('Convergence')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)

plt.suptitle('Fuzzy C-Means — Iris Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('vis_fcm_iris.png', dpi=150, bbox_inches='tight')
print("Saved vis_fcm_iris.png")

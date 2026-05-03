"""Live visualization utility for JAX clustering models."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class LiveVisualizer:
    """
    Reusable live visualization for clustering models.

    Automatically reduces high-dimensional data to 2D using PCA
    and displays cluster formation during training.

    Args:
        pause_time: Seconds to pause between iterations (default: 0.15)
        show_confidence: Show confidence/membership values as point sizes (default: True)
        show_pca_variance: Show PCA variance explained in title (default: True)
        save_path: Path to save final plot (default: None, no saving)
    """

    def __init__(
        self,
        pause_time: float = 0.15,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_path: str = None
    ):
        self.pause_time = pause_time
        self.show_confidence = show_confidence
        self.show_pca_variance = show_pca_variance
        self.save_path = save_path
        self.X_2d = None
        self.pca = None
        self.fig = None
        self.ax = None
        self._initialized = False
        self._title = None

    def setup(self, X: np.ndarray, title: str = "Clustering"):
        """Initialize visualization with data."""
        self._title = title

        if X.shape[1] > 2:
            self.pca = PCA(n_components=2)
            self.X_2d = self.pca.fit_transform(X)
        else:
            self.X_2d = X

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Set up title with PCA variance if applicable
        full_title = title
        if self.pca and self.show_pca_variance:
            variance = self.pca.explained_variance_ratio_
            full_title += f"\n(PCA: {variance[0]:.1%} + {variance[1]:.1%} = {variance.sum():.1%} variance)"

        self.ax.set_xlabel('PC1' if self.pca else 'Feature 1')
        self.ax.set_ylabel('PC2' if self.pca else 'Feature 2')
        self.ax.set_title(full_title)
        self._initialized = True
        plt.show()

    def update(self, centroids: np.ndarray, labels: np.ndarray,
               weights: np.ndarray, iteration: int, objective: float,
               max_iter: int, **kwargs):
        """
        Update visualization with current state.

        Args:
            centroids: Cluster centers (c, d)
            labels: Cluster assignments (n,)
            weights: Point weights for sizing (n,) - membership/typicality
            iteration: Current iteration
            objective: Current objective value
            max_iter: Maximum iterations
            **kwargs: Additional info (e.g., outlier_count)
        """
        if not self._initialized:
            return

        self.ax.clear()

        # Transform centroids to 2D
        centroids_2d = self.pca.transform(centroids) if self.pca else centroids

        # Determine point sizes based on show_confidence option
        if self.show_confidence:
            point_sizes = 100 * weights
            alpha = 0.6
        else:
            point_sizes = 50  # Fixed size
            alpha = 0.7

        # Plot points
        scatter = self.ax.scatter(
            self.X_2d[:, 0], self.X_2d[:, 1],
            c=labels, s=point_sizes,
            cmap='viridis', alpha=alpha,
            edgecolors='black', linewidths=0.5
        )

        # Plot centroids
        self.ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            marker='v', s=300, c='red',
            edgecolors='black', linewidths=2,
            label='Centroids', zorder=5
        )

        # Annotations
        info_text = f'Iteration: {iteration}/{max_iter}\nObjective: {objective:.4f}'
        if self.show_confidence:
            avg_conf = np.mean(weights)
            info_text += f'\nAvg Confidence: {avg_conf:.3f}'
        if 'outlier_count' in kwargs:
            info_text += f'\nOutliers: {kwargs["outlier_count"]}'

        self.ax.text(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Update title with PCA variance if applicable
        full_title = self._title
        if self.pca and self.show_pca_variance:
            variance = self.pca.explained_variance_ratio_
            full_title += f"\n(PCA: {variance[0]:.1%} + {variance[1]:.1%} = {variance.sum():.1%} variance)"

        self.ax.set_xlabel('PC1' if self.pca else 'Feature 1')
        self.ax.set_ylabel('PC2' if self.pca else 'Feature 2')
        self.ax.set_title(full_title)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)

        plt.draw()
        plt.pause(self.pause_time)

    def close(self):
        """Close visualization and save plot if requested."""
        if self._initialized:
            if self.save_path:
                self.fig.savefig(self.save_path, dpi=300, bbox_inches='tight')
            plt.ioff()
            plt.close(self.fig)
            self._initialized = False

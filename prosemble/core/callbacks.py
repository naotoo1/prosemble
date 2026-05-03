"""Callback system for clustering model training."""

import numpy as np


class Callback:
    """Base class for training callbacks.

    Subclass this and override methods to hook into the training loop.
    All methods are no-ops by default.
    """

    def on_fit_start(self, model, X):
        """Called once before training begins.

        Parameters
        ----------
        model : FuzzyClusteringBase
            The model being trained
        X : array-like
            Training data
        """
        pass

    def on_iteration_end(self, model, info):
        """Called after each iteration.

        Parameters
        ----------
        model : FuzzyClusteringBase
            The model being trained
        info : dict
            Standardized iteration info with keys:
            - centroids: np.ndarray (n_clusters, n_features)
            - labels: np.ndarray (n_samples,) int
            - weights: np.ndarray (n_samples,) float
            - iteration: int
            - objective: float
            - max_iter: int
        """
        pass

    def on_fit_end(self, model, info):
        """Called once after training ends.

        Parameters
        ----------
        model : FuzzyClusteringBase
            The model being trained
        info : dict
            Same format as on_iteration_end
        """
        pass


class VisualizationCallback(Callback):
    """Live visualization during clustering training.

    Wraps LiveVisualizer and translates callback events into
    visualizer calls with correctly formatted arguments.

    Parameters
    ----------
    pause_time : float, default=0.15
        Seconds to pause between updates
    show_confidence : bool, default=True
        Show confidence values as point sizes
    show_pca_variance : bool, default=True
        Show PCA variance explained in title
    save_path : str, optional
        Path to save final plot
    """

    def __init__(self, pause_time=0.15, show_confidence=True,
                 show_pca_variance=True, save_path=None):
        from prosemble.core.visualizer import LiveVisualizer
        self.visualizer = LiveVisualizer(
            pause_time, show_confidence, show_pca_variance, save_path
        )

    def on_fit_start(self, model, X):
        X_np = np.asarray(X)
        title = model.__class__.__name__ + " Training"
        self.visualizer.setup(X_np, title)

    def on_iteration_end(self, model, info):
        self.visualizer.update(**info)

    def on_fit_end(self, model, info):
        self.visualizer.update(**info)
        self.visualizer.close()

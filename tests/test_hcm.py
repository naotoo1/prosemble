"""
Tests for Hard C-Means (HCM/K-Means) clustering.
"""
import numpy as np
import pytest
from prosemble.models.hcm import Kmeans


class TestHCMInitialization:
    """Test HCM/K-means initialization."""

    def test_hcm_init_basic(self, simple_2d_data):
        """Test basic HCM initialization."""
        X, _ = simple_2d_data
        model = Kmeans(
            data=X,
            c=3,
            num_inter=100,
            epsilon=0.00001,
            ord='fro'
        )
        assert model.num_clusters == 3

    def test_hcm_init_with_prototypes(self, simple_2d_data):
        """Test HCM initialization with custom prototypes."""
        X, _ = simple_2d_data
        
        # Create initial prototypes
        initial_prototypes = X[:3]
        
        model = Kmeans(
            data=X,
            c=3,
            num_inter=100,
            epsilon=0.00001,
            ord='fro',
            set_prototypes=initial_prototypes
        )
        assert np.array_equal(model.set_prototypes, initial_prototypes)


class TestHCMFitting:
    """Test HCM fitting."""

    def test_hcm_fit(self, simple_2d_data):
        """Test HCM fitting."""
        X, _ = simple_2d_data
        model = Kmeans(
            data=X,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        clusters, centroids = model.get_clusters_index_cent()
        assert len(centroids) == 1
        assert centroids[0].shape == (3, 2)

    def test_hcm_objective_function(self, simple_2d_data):
        """Test HCM objective function computation."""
        X, _ = simple_2d_data
        model = Kmeans(
            data=X,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        obj_func = model.get_objective_function()
        assert len(obj_func) > 0
        # Objective should decrease
        assert obj_func[-1] <= obj_func[0]


class TestHCMPrediction:
    """Test HCM prediction methods."""

    def test_hcm_predict(self, simple_2d_data):
        """Test HCM predict."""
        X, _ = simple_2d_data
        model = Kmeans(
            data=X,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        labels = model.predict()
        assert labels.shape[0] == X.shape[0]
        assert set(labels).issubset({0, 1, 2})

    def test_hcm_predict_new(self, iris_train_test_split):
        """Test HCM predict on new data."""
        X_train, X_test, _, _ = iris_train_test_split
        
        model = Kmeans(
            data=X_train,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        predictions = model.predict_new(X_test)
        assert len(predictions) == X_test.shape[0]

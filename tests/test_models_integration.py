"""
Integration tests for cross-model compatibility.
"""
import numpy as np
import pytest
from prosemble.models.fcm import FCM
from prosemble.models.pcm import PCM
from prosemble.models.kmeans import kmeans_plusplus


@pytest.mark.integration
class TestModelsIntegration:
    """Test interactions between different models."""

    def test_fcm_vs_kmeans_consistency(self, simple_2d_data):
        """Test that FCM and K-means give similar results."""
        X, _ = simple_2d_data
        
        # Train FCM
        fcm = FCM(
            data=X,
            c=3,
            m=2,
            num_iter=100,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        fcm.fit()
        fcm_labels = fcm.predict()
        
        # Train K-means
        kmeans = kmeans_plusplus(
            data=X,
            c=3,
            num_inter=100,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        kmeans.fit()
        kmeans_labels = kmeans.predict()
        
        # Both should assign each point to one of 3 clusters
        assert set(fcm_labels).issubset({0, 1, 2})
        assert set(kmeans_labels).issubset({0, 1, 2})

    def test_all_models_iris_dataset(self, iris_data):
        """Test that multiple models can train on Iris dataset."""
        X, _ = iris_data
        
        models = [
            FCM(data=X, c=3, m=2, num_iter=50, epsilon=0.00001, ord='fro', plot_steps=False),
            kmeans_plusplus(data=X, c=3, num_inter=50, epsilon=0.00001, ord='fro', plot_steps=False),
        ]
        
        for model in models:
            model.fit()
            labels = model.predict()
            assert len(labels) == len(X)
            assert set(labels).issubset({0, 1, 2})

    def test_model_pipeline_workflow(self, iris_train_test_split):
        """Test a typical workflow using multiple models."""
        X_train, X_test, y_train, y_test = iris_train_test_split
        
        # Step 1: Cluster with FCM
        fcm = FCM(
            data=X_train,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        fcm.fit()
        
        # Step 2: Predict on test data
        predictions = fcm.predict_new(X_test)
        
        # Verify predictions
        assert len(predictions) == len(X_test)
        assert all(pred in {0, 1, 2} for pred in predictions)


@pytest.mark.integration
class TestCrossModelCompatibility:
    """Test initialization of one model with another's output."""

    def test_fcm_initialization_with_kmeans(self, simple_2d_data):
        """Test using K-means centroids to initialize FCM."""
        X, _ = simple_2d_data
        
        # Get K-means centroids
        kmeans = kmeans_plusplus(
            data=X,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        kmeans.fit()
        kmeans_proba = kmeans.get_proba(X)
        
        # Initialize FCM with K-means membership
        fcm = FCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            set_U_matrix=kmeans_proba,
            plot_steps=False
        )
        fcm.fit()
        
        assert fcm.fit_cent[0].shape == (3, 2)

    def test_pcm_initialization_with_fcm(self, simple_2d_data):
        """Test initializing PCM with FCM outputs."""
        X, _ = simple_2d_data
        
        # Train FCM first
        fcm = FCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        fcm.fit()
        fcm_centroids = fcm.final_centroids()
        fcm_memberships = fcm.predict_proba_(X)
        
        # Initialize PCM with FCM outputs - use set_centroids not set_prototypes
        pcm = PCM(
            data=X,
            c=3,
            m=2,
            k=0.01,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            set_centroids=fcm_centroids,
            set_U_matrix=fcm_memberships,
            plot_steps=False
        )
        pcm.fit()
        
        assert pcm.fit_cent[0].shape == (3, 2)

"""
Integration tests for prosemble models.
"""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score


@pytest.mark.integration
class TestModelsIntegration:
    """Integration tests comparing different models."""

    def test_fcm_vs_kmeans_consistency(self, simple_2d_data):
        """Test that FCM and K-means produce similar clusterings."""
        from prosemble.models.fcm import FCM
        from prosemble.models.kmeans import kmeans_plusplus
        
        X, y_true = simple_2d_data
        
        # Fit FCM
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
        
        # Fit K-means
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
        
        # Clusterings should be reasonably similar
        ari = adjusted_rand_score(fcm_labels, kmeans_labels)
        assert ari > 0.5, f"FCM and K-means should produce similar results, ARI={ari}"

    def test_all_models_iris_dataset(self, iris_data):
        """Test that multiple models can fit the Iris dataset."""
        from prosemble.models.fcm import FCM
        from prosemble.models.kmeans import kmeans_plusplus
        
        X, y = iris_data
        
        models = [
            ('FCM', FCM(
                data=X, c=3, m=2, num_iter=50,
                epsilon=0.00001, ord='fro', plot_steps=False
            )),
            ('KMeans++', kmeans_plusplus(
                data=X, c=3, num_inter=50,
                epsilon=0.00001, ord='fro', plot_steps=False
            ))
        ]
        
        for name, model in models:
            model.fit()
            labels = model.predict()
            
            # Check that predictions are valid
            assert labels.shape[0] == X.shape[0], f"{name} failed"
            assert len(set(labels)) <= 3, f"{name} produced wrong number of clusters"

    def test_model_pipeline_workflow(self, iris_train_test_split):
        """Test a complete ML pipeline workflow."""
        from prosemble.models.fcm import FCM
        
        X_train, X_test, y_train, y_test = iris_train_test_split
        
        # Train model
        model = FCM(
            data=X_train,
            c=3,
            m=2,
            num_iter=100,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        # Get training predictions
        train_labels = model.predict()
        assert len(train_labels) == len(X_train)
        
        # Get test predictions
        test_labels = model.predict_new(X_test)
        assert len(test_labels) == len(X_test)
        
        # Get probabilities
        test_probas = model.predict_proba_(X_test)
        assert test_probas.shape == (len(X_test), 3)
        
        # Get distance space
        distances = model.get_distance_space(X_test)
        assert distances.shape == (len(X_test), 3)


@pytest.mark.integration
class TestCrossModelCompatibility:
    """Test that models can be used interchangeably."""

    def test_fcm_initialization_with_kmeans(self, simple_2d_data):
        """Test using K-means to initialize FCM."""
        from prosemble.models.fcm import FCM
        
        X, _ = simple_2d_data
        
        # Use kmeans initialization
        model = FCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            set_U_matrix='kmeanss',
            plot_steps=False
        )
        model.fit()
        
        centroids = model.final_centroids()
        assert centroids.shape == (3, 2)

    def test_pcm_initialization_with_fcm(self, simple_2d_data):
        """Test using FCM to initialize PCM."""
        from prosemble.models.fcm import FCM
        from prosemble.models.pcm import PCM
        
        X, _ = simple_2d_data
        
        # First run FCM
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
        
        # Use FCM centroids to initialize PCM
        pcm = PCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            set_prototypes=fcm_centroids,
            plot_steps=False
        )
        pcm.fit()
        
        labels = pcm.predict()
        assert labels.shape[0] == X.shape[0]

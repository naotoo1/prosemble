"""
Tests for K-Nearest Neighbors (KNN).
"""
import numpy as np
import pytest
from prosemble.models.knn import KNN


class TestKNNBasic:
    """Basic tests for KNN."""

    def test_knn_initialization(self, iris_train_test_split):
        """Test KNN initialization."""
        X_train, _, y_train, _ = iris_train_test_split
        
        model = KNN(dataset=X_train, labels=y_train, c=3)
        assert model.neighbours == 3

    def test_knn_fit_predict(self, iris_train_test_split):
        """Test KNN predict."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = KNN(dataset=X_train, labels=y_train, c=5)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset(set(y_train))

    def test_knn_different_k_values(self, iris_train_test_split):
        """Test KNN with different k values."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        k_values = [1, 3, 5, 7]
        
        for k in k_values:
            model = KNN(dataset=X_train, labels=y_train, c=k)
            predictions = model.predict(X_test)
            
            assert len(predictions) == len(X_test)

    def test_knn_single_neighbor(self, iris_train_test_split):
        """Test KNN with k=1."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = KNN(dataset=X_train, labels=y_train, c=1)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_knn_get_proba(self, iris_train_test_split):
        """Test KNN probability predictions."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = KNN(dataset=X_train, labels=y_train, c=5)
        
        # Test on a few samples
        probas = model.get_proba(X_test[:5])
        assert len(probas) == 5
        # All probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in probas)

    def test_knn_distance_space(self, iris_train_test_split):
        """Test KNN distance space computation."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = KNN(dataset=X_train, labels=y_train, c=5)
        
        # Test distance space for single sample
        distances = model.distance_space(X_test[0])
        assert len(distances) == len(X_train)
        assert np.all(distances >= 0)  # Distances should be non-negative
        # Distances should be sorted
        assert np.all(distances[:-1] <= distances[1:])

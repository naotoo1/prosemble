"""
Tests for K-Nearest Neighbors (KNN).
Note: This tests sklearn's KNN as prosemble doesn't have a custom KNN implementation.
"""
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier


class TestKNNBasic:
    """Basic tests for KNN using sklearn."""

    def test_knn_initialization(self):
        """Test KNN initialization."""
        model = KNeighborsClassifier(n_neighbors=3)
        assert model.n_neighbors == 3

    def test_knn_fit_predict(self, iris_train_test_split):
        """Test KNN fit and predict."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset(set(y_train))

    def test_knn_different_k_values(self, iris_train_test_split):
        """Test KNN with different k values."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        k_values = [1, 3, 5, 7]
        
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert len(predictions) == len(X_test)

    def test_knn_single_neighbor(self, iris_train_test_split):
        """Test KNN with k=1."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

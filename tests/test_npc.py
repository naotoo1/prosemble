"""
Tests for Nearest Prototype Classifier (NPC).
"""
import numpy as np
import pytest
from prosemble.models.npc import NPC


class TestNPCBasic:
    """Basic tests for NPC."""

    def test_npc_initialization(self, iris_data):
        """Test NPC initialization."""
        X, y = iris_data
        
        # Create prototypes (use class means)
        unique_classes = np.unique(y)
        prototypes = np.array([X[y == c].mean(axis=0) for c in unique_classes])
        
        model = NPC(prototypes=prototypes, y_labels=unique_classes)
        assert model.prototypes.shape == (3, 4)

    def test_npc_predict(self, iris_train_test_split):
        """Test NPC prediction."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        # Create prototypes from training data
        unique_classes = np.unique(y_train)
        prototypes = np.array([X_train[y_train == c].mean(axis=0) for c in unique_classes])
        
        model = NPC(prototypes=prototypes, y_labels=unique_classes)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset(set(unique_classes))

    def test_npc_single_sample(self, iris_data):
        """Test NPC with single sample prediction."""
        X, y = iris_data
        
        unique_classes = np.unique(y)
        prototypes = np.array([X[y == c].mean(axis=0) for c in unique_classes])
        
        model = NPC(prototypes=prototypes, y_labels=unique_classes)
        
        # Predict single sample
        single_sample = X[0:1]
        prediction = model.predict(single_sample)
        
        assert len(prediction) == 1
        assert prediction[0] in unique_classes

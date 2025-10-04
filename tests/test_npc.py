"""
Tests for Nearest Prototype Classifier (NPC1).
"""
import numpy as np
import pytest
from prosemble.models.npc import NPC1


class TestNPC1Basic:
    """Basic tests for NPC1."""

    def test_npc1_initialization(self, iris_train_test_split):
        """Test NPC1 initialization."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = NPC1(
            data=X_train,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            labels=y_train,
            plot_steps=False
        )
        assert model.num_clusters == 3

    def test_npc1_fit_predict(self, iris_train_test_split):
        """Test NPC1 fit and predict."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = NPC1(
            data=X_train,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            labels=y_train,
            plot_steps=False
        )
        model.fit()
        
        predictions = model.predict_sample(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset(set(y_train))

    def test_npc1_prototypes(self, iris_train_test_split):
        """Test NPC1 prototypes retrieval."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = NPC1(
            data=X_train,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            labels=y_train,
            plot_steps=False
        )
        model.fit()
        
        prototypes = model.prototypes()
        assert prototypes.shape[0] == 3
        assert prototypes.shape[1] == X_train.shape[1]

    def test_npc1_predict_proba(self, iris_train_test_split):
        """Test NPC1 probability predictions."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = NPC1(
            data=X_train,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            labels=y_train,
            plot_steps=False
        )
        model.fit()
        
        probas = model.get_predict_proba(X_test[:5])
        assert probas.shape == (5, 3)
        # Probabilities should sum to approximately 1
        np.testing.assert_array_almost_equal(
            probas.sum(axis=1),
            np.ones(5),
            decimal=5
        )

    def test_npc1_with_custom_prototypes(self, iris_train_test_split):
        """Test NPC1 with custom prototypes."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        # Create custom prototypes (use class means)
        unique_classes = np.unique(y_train)
        custom_prototypes = np.array([X_train[y_train == c].mean(axis=0) for c in unique_classes])
        
        model = NPC1(
            data=X_train,
            c=3,
            num_inter=50,
            epsilon=0.00001,
            ord='fro',
            labels=y_train,
            set_prototypes_=custom_prototypes,
            plot_steps=False
        )
        model.fit()
        
        predictions = model.predict_sample(X_test)
        assert len(predictions) == len(X_test)

    def test_npc1_invalid_opt_metric(self, iris_train_test_split):
        """Test NPC1 with invalid opt_metric type."""
        X_train, _, y_train, _ = iris_train_test_split
        
        with pytest.raises(TypeError, match="opt_metric must be a float"):
            NPC1(
                data=X_train,
                c=3,
                num_inter=50,
                epsilon=0.00001,
                ord='fro',
                labels=y_train,
                opt_metric="invalid",
                plot_steps=False
            )

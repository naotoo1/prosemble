# Testing Guide for Prosemble

This document provides comprehensive information about testing in the prosemble project.

## Test Coverage Summary

The test suite now includes:

### Models Tested (21 total)

#### Fuzzy Clustering Models
- ✅ **FCM** (Fuzzy C-Means) - Comprehensive tests
- ✅ **AFCM** (Adaptive Fuzzy C-Means) - Basic tests
- ✅ **FPCM** (Fuzzy Possibilistic C-Means) - Basic tests
- ✅ **PFCM** (Possibilistic Fuzzy C-Means) - Basic tests
- ✅ **PCM** (Possibilistic C-Means) - Basic tests

#### Kernel-Based Models
- ✅ **KFCM** (Kernel Fuzzy C-Means) - Basic tests
- ✅ **KAFCM** (Kernel Adaptive Fuzzy C-Means) - Basic tests
- ✅ **KPCM** (Kernel Possibilistic C-Means) - Basic tests
- ⚠️ **KFPCM** (Kernel Fuzzy Possibilistic C-Means) - Not yet tested
- ⚠️ **KPFCM** (Kernel Possibilistic Fuzzy C-Means) - Not yet tested
- ⚠️ **KIPCM** (Kernel Improved Possibilistic C-Means) - Not yet tested

#### Other Models
- ✅ **HCM/K-Means** (Hard C-Means) - Basic tests
- ✅ **K-Means++** - Comprehensive tests
- ✅ **KNN** (K-Nearest Neighbors) - Basic tests
- ✅ **NPC** (Nearest Prototype Classifier) - Basic tests
- ✅ **SOM** (Self-Organizing Map) - Basic tests
- ✅ **BGPC** (Bounded Gradient Projection Clustering) - Basic tests
- ⚠️ **IPCM** (Improved Possibilistic C-Means) - Not yet tested
- ⚠️ **IPCM2** (Improved Possibilistic C-Means v2) - Not yet tested

### Test Categories

1. **Unit Tests** - Test individual model components
   - Initialization
   - Fitting
   - Prediction
   - Utility methods

2. **Integration Tests** - Test cross-model compatibility
   - Model pipelines
   - Model initialization chains (e.g., FCM → PCM)
   - Consistency between models

3. **Edge Case Tests** - Test boundary conditions
   - Single cluster
   - Many clusters
   - Different input types
   - Invalid parameters

4. **Performance Tests** - Test model behavior
   - Convergence
   - Reproducibility
   - Objective function monotonicity

## Test Statistics

- **Total test files**: 13
- **Models with comprehensive tests**: 2 (FCM, K-means++)
- **Models with basic tests**: 11
- **Models without tests**: 8
- **Estimated coverage**: ~60-70% (after running all tests)

## Installation for Testing

```bash
# Install package in development mode
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt
```

## Running Tests

### Quick Start
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=prosemble --cov-report=term-missing
```

### Advanced Usage
```bash
# Run specific test file
pytest tests/test_fcm.py

# Run specific test class
pytest tests/test_fcm.py::TestFCMFitting

# Run specific test
pytest tests/test_fcm.py::TestFCMFitting::test_fcm_fit

# Run tests matching pattern
pytest -k "fcm or kmeans"

# Run tests in parallel (faster)
pytest -n auto

# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=prosemble --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows

# Generate XML coverage report (for CI)
pytest --cov=prosemble --cov-report=xml
```

## Test Organization

### File Naming
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure Example
```python
class TestModelName:
    """Test ModelName clustering/classification."""

    def test_initialization(self, fixture):
        """Test model initialization."""
        # Arrange
        X, y = fixture
        
        # Act
        model = ModelName(data=X, c=3)
        
        # Assert
        assert model.num_clusters == 3

    def test_fit(self, fixture):
        """Test model fitting."""
        X, y = fixture
        model = ModelName(data=X, c=3)
        
        # Should not raise
        model.fit()
        
    def test_predict(self, fixture):
        """Test prediction."""
        X, y = fixture
        model = ModelName(data=X, c=3)
        model.fit()
        
        labels = model.predict()
        
        assert labels.shape[0] == X.shape[0]
        assert set(labels).issubset({0, 1, 2})
```

## Common Fixtures

Defined in `tests/conftest.py`:

```python
iris_data                # Iris dataset (X, y)
iris_train_test_split    # Split Iris dataset
simple_2d_data           # Simple 2D clustered data
simple_3d_data           # Simple 3D clustered data
random_data              # Random data for edge cases
```

## Continuous Integration

### GitHub Actions Workflow

Tests run automatically on:
- Push to main, develop, or feature branches
- Pull requests to main or develop
- Multiple platforms: Ubuntu, macOS, Windows
- Multiple Python versions: 3.9, 3.10, 3.11, 3.12

### Coverage Threshold

Minimum coverage threshold: **70%**

CI will fail if coverage drops below this threshold.

## Adding New Tests

### Checklist for New Model Tests

- [ ] Create `tests/test_<model_name>.py`
- [ ] Test initialization
  - [ ] Basic initialization
  - [ ] With custom parameters
  - [ ] Invalid parameters (should raise)
- [ ] Test fitting
  - [ ] Basic fit
  - [ ] Convergence check
  - [ ] Objective function
- [ ] Test prediction
  - [ ] Predict on training data
  - [ ] Predict on new data
  - [ ] Predict probabilities (if applicable)
- [ ] Test edge cases
  - [ ] Single cluster
  - [ ] Many clusters
  - [ ] Different input types
- [ ] Test utility methods
  - [ ] Get centroids/prototypes
  - [ ] Get distance space
  - [ ] Other model-specific methods

## Current Gaps

Models still needing tests:

1. **IPCM** - Improved Possibilistic C-Means
2. **IPCM2** - Improved Possibilistic C-Means v2
3. **KFPCM** - Kernel Fuzzy Possibilistic C-Means
4. **KPFCM** - Kernel Possibilistic Fuzzy C-Means
5. **KIPCM** - Kernel Improved Possibilistic C-Means
6. **KIPCM2** - Kernel Improved Possibilistic C-Means v2

## Next Steps

1. ✅ Add tests for remaining 6 models
2. ✅ Increase coverage to >80%
3. ✅ Add performance benchmarks
4. ✅ Add property-based tests (hypothesis)
5. ✅ Add mutation testing

## Troubleshooting

### Import Errors
```bash
# Make sure package is installed in development mode
pip install -e .
```

### Slow Tests
```bash
# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

### Test Failures
```bash
# Run with verbose output
pytest -vv

# Stop on first failure for debugging
pytest -x

# Re-run only failed tests
pytest --lf
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)

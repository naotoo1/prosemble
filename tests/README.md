# Prosemble Test Suite

Comprehensive test suite for the prosemble package.

## Overview

This test suite provides comprehensive coverage for all prosemble models and utilities, including:

- **Unit tests** for individual model components
- **Integration tests** for cross-model compatibility
- **Edge case tests** for boundary conditions
- **Performance tests** for reproducibility and convergence

## Test Structure

```
tests/
├── conftest.py                    # Shared pytest fixtures
├── test_core.py                   # Core utilities tests
├── test_dataset.py                # Dataset utilities tests
├── test_fcm.py                    # Fuzzy C-Means tests
├── test_kmeans.py                 # K-means++ tests
├── test_pcm.py                    # Possibilistic C-Means tests
├── test_hcm.py                    # Hard C-Means tests
├── test_fuzzy_models.py           # FPCM, PFCM, AFCM tests
├── test_kernel_models.py          # KFCM, KPCM, KAFCM tests
├── test_npc.py                    # Nearest Prototype Classifier tests
├── test_som.py                    # Self-Organizing Map tests
├── test_knn.py                    # K-Nearest Neighbors tests
├── test_bgpc.py                   # BGPC tests
└── test_models_integration.py     # Cross-model integration tests
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with coverage report
```bash
pytest tests/ --cov=prosemble --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_fcm.py -v
```

### Run specific test class
```bash
pytest tests/test_fcm.py::TestFCMFitting -v
```

### Run specific test method
```bash
pytest tests/test_fcm.py::TestFCMFitting::test_fcm_fit -v
```

### Run tests excluding slow tests
```bash
pytest tests/ -m "not slow"
```

### Run only integration tests
```bash
pytest tests/ -m integration
```

### Run tests in parallel (faster)
```bash
pytest tests/ -n auto
```

## Test Markers

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests

## Coverage Goals

- **Overall coverage**: >70%
- **Core modules**: >80%
- **Model modules**: >75%

## Test Fixtures

Common fixtures available in all tests (defined in `conftest.py`):

- `iris_data` - Iris dataset (X, y)
- `iris_train_test_split` - Split Iris dataset
- `simple_2d_data` - Simple 2D clustered data
- `simple_3d_data` - Simple 3D clustered data
- `random_data` - Random data for edge cases

## Writing New Tests

### Test Template

```python
import numpy as np
import pytest
from prosemble.models.your_model import YourModel


class TestYourModelInitialization:
    """Test model initialization."""

    def test_basic_init(self, simple_2d_data):
        """Test basic initialization."""
        X, _ = simple_2d_data
        model = YourModel(data=X, c=3)
        assert model.num_clusters == 3


class TestYourModelFitting:
    """Test model fitting."""

    def test_fit(self, simple_2d_data):
        """Test model fitting."""
        X, _ = simple_2d_data
        model = YourModel(data=X, c=3)
        model.fit()
        # Add assertions


class TestYourModelPrediction:
    """Test model prediction."""

    def test_predict(self, simple_2d_data):
        """Test prediction."""
        X, _ = simple_2d_data
        model = YourModel(data=X, c=3)
        model.fit()
        labels = model.predict()
        assert len(labels) == len(X)
```

## Best Practices

1. **Use descriptive test names** - Test names should describe what they test
2. **One assertion per test** - Keep tests focused
3. **Use fixtures** - Reuse common setup code
4. **Test edge cases** - Include boundary conditions
5. **Mock slow operations** - Keep tests fast when possible
6. **Document expected behavior** - Use docstrings

## Continuous Integration

Tests are automatically run on:
- Every push to main/develop branches
- Every pull request
- Multiple OS: Ubuntu, macOS, Windows
- Multiple Python versions: 3.9, 3.10, 3.11, 3.12

## Test Coverage Report

After running tests with coverage, view the HTML report:

```bash
pytest tests/ --cov=prosemble --cov-report=html
open htmlcov/index.html  # On macOS/Linux
start htmlcov/index.html  # On Windows
```

## Troubleshooting

### Tests fail with import errors
```bash
pip install -e .
```

### Tests are slow
```bash
pytest tests/ -n auto  # Run in parallel
pytest tests/ -m "not slow"  # Skip slow tests
```

### Coverage is low
- Add more test cases
- Test edge cases
- Add integration tests

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >70% coverage
4. Update this README if needed

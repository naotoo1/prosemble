"""Shared pytest configuration."""

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope='module')
def _clear_jax_caches():
    """Drop JAX's compiled functions after each test module.

    Every model compiles its own training step, and JAX keeps each
    compiled function for the life of the process. Without this the memory
    of a test process grows through the whole suite, which runs the
    parallel workers out of memory.
    """
    yield
    jax.clear_caches()
    gc.collect()

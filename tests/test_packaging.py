"""Packaging checks: version metadata and declared dependencies."""

import tomllib
from importlib.metadata import requires
from pathlib import Path

import prosemble

_PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'


def _project():
    with open(_PYPROJECT, 'rb') as f:
        return tomllib.load(f)['project']


def test_version_matches_pyproject():
    assert prosemble.__version__ == _project()['version']


def test_jax_stack_is_a_core_dependency():
    """``import prosemble`` needs JAX, so a plain install must provide it."""
    core = {
        r.split(';')[0].strip()
        for r in requires('prosemble')
        if 'extra ==' not in r
    }
    names = {r.split('>')[0].split('=')[0].split('[')[0].strip() for r in core}
    assert {'jax', 'jaxlib', 'chex', 'optax'} <= names

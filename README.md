# Prosemble

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-1.0.0-orange.svg)](https://pypi.org/project/prosemble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/naotoo1/prosemble/actions/workflows/ci.yml/badge.svg)](https://github.com/naotoo1/prosemble/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/prosemble/badge/?version=latest)](https://prosemble.readthedocs.io/en/latest/)

## Description

This is a JAX-based Python toolbox for research and application of prototype-based machine learning methods and other interpretable models. All models are JIT-compiled and run on CPU, GPU and TPU. The focus of Prosemble is ease-of-use, extensibility and speed.

## Installation

Prosemble can be installed using pip:

```bash
pip install prosemble
```

To install with JAX support:

```bash
pip install "prosemble[jax]"         # CPU
pip install "prosemble[jax-cuda12]"  # GPU (CUDA 12)
```

To install the development version:

```bash
git clone https://github.com/naotoo1/prosemble.git
cd prosemble
pip install -e ".[all]"
```

## Documentation

The full documentation is available at [prosemble.readthedocs.io](https://prosemble.readthedocs.io).

To build locally:

```bash
cd sphinx-docs && make html
```

## Examples

See the [examples/](examples/) directory.

## Development

Prosemble provides a reproducible development environment using [devenv](https://devenv.sh/getting-started/):

```bash
git clone https://github.com/naotoo1/prosemble.git
cd prosemble
devenv shell
uv sync --extra jax --extra dev
uv run pytest tests/ -x -q
```

## Citation

```bibtex
@misc{Otoo_Prosemble_2022,
  author       = {Otoo, Nana Abeka},
  title        = {Prosemble},
  year         = {2022},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/naotoo1/Prosemble}},
}
```

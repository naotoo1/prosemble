# Prosemble

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![GitHub Version](https://img.shields.io/badge/version-0.9.2-yellow.svg)](https://github.com/naotoo1/Prosemble)
[![PyPI Version](https://img.shields.io/badge/pypi-0.9.2-orange.svg)](https://pypi.org/project/prosemble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

Prosemble is a Python library for prototype-based machine learning models.

## Installation

Prosemble can be installed using pip:

```bash
pip install prosemble
```

If you have installed Prosemble before and want to upgrade to the latest version, you can run the following command in your terminal:

```bash
pip install -U prosemble
```

To install the latest development version directly from the GitHub repository:

```bash
pip install git+https://github.com/naotoo1/prosemble
```

Prosemble provides a fully reproducible development environment using Nix and [devenv](https://devenv.sh/getting-started/). Once you have installed Nix and devenv, you can do the following:

   ```bash
   mkdir -p ~/.config/nix
   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
   nix profile install --accept-flake-config "github:cachix/devenv/latest"
   ```

Activate the reproducible development envirnment with:
   ```bash
   devenv shell
   ```

Install Prosemble in develop mode with:
   ```bash
   devenv shell -- uv pip install -e .[all]
   ```

You may optionally consider using [direnv](https://direnv.net/) for automatic shell activation when entering the project directory.


## Citation

If you use Prosemble in your research, please cite it using the following BibTeX entry:

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

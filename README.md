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
## Development Environment
Prosemble provides a fully reproducible development environment using Nix and [devenv](https://devenv.sh/getting-started/). Once you have installed Nix and devenv, you can do the following:

   ```bash
   mkdir -p ~/.config/nix
   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
   nix profile install --accept-flake-config "github:cachix/devenv/latest"
   ```

Then clone and enter the project directory:

```bash
git clone https://github.com/naotoo1/prosemble.git
cd prosemble
```

Activate the reproducible development environment with:
   ```bash
   devenv shell
   ```
You may optionally consider using [direnv](https://direnv.net/) for automatic shell activation when entering the project directory.

To install Prosemble in development mode, follow these steps to set up your environment with all the necessary dependencies while ensuring the package is installed with live code editing capabilities. To run the local reproducible development environment, execute the following lock file commands:

```bash
# Generate requirements file
generate-requirements

# Update lock files
update-lock-files

# Install dependencies from lock file
install-from-lock
```
Alternatively, use this one-liner:

```bash
setup-python-env
```
To run Prosemble inside a reproducible Docker container, execute:
```bash
# Build the Docker container
create-cpu-container
# Run the container 
run-cpu-container
```
When working with Prosemble in development mode, changes to the code take effect immediately without reinstallation. Use ```git pull``` to get the latest updates from the repository. Run tests after making changes to verify functionality


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

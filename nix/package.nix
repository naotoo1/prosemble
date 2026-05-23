# Nixpkgs expression for prosemble
#
# To submit to nixpkgs:
# 1. Copy this file to pkgs/development/python-modules/prosemble/default.nix
# 2. Add to pkgs/top-level/python-packages.nix:
#      prosemble = callPackage ../development/python-modules/prosemble { };
# 3. Update the hash after publishing to PyPI:
#      nix-prefetch-url --unpack https://github.com/naotoo1/prosemble/archive/refs/tags/v2.2.0.tar.gz
# 4. PR title: python3Packages.prosemble: init at 2.2.0
{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  # Build
  setuptools,
  setuptools-scm,
  # Runtime
  numpy,
  scikit-learn,
  matplotlib,
  # Optional (JAX)
  jax,
  jaxlib,
  chex,
  optax,
  # Tests
  pytestCheckHook,
}:

buildPythonPackage (finalAttrs: {
  pname = "prosemble";
  version = "2.2.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "naotoo1";
    repo = "prosemble";
    tag = "v${finalAttrs.version}";
    hash = "sha256-296Z1lHBcOVQbkfO7KlNnWe7f7xjbKFHJRfGUPaHKWY=";
  };

  build-system = [
    setuptools
    setuptools-scm
  ];

  dependencies = [
    numpy
    scikit-learn
    matplotlib
  ];

  optional-dependencies = {
    jax = [
      jax
      jaxlib
      chex
      optax
    ];
  };

  nativeCheckInputs = [
    pytestCheckHook
    jax
    jaxlib
    chex
    optax
  ];

  disabledTestPaths = [
    # Requires geomstats which is not packaged
    "tests/test_riemannian_neural_gas.py"
  ];

  pythonImportsCheck = [
    "prosemble"
    "prosemble.models"
    "prosemble.core"
  ];

  meta = {
    description = "JAX-based prototype machine learning library";
    homepage = "https://github.com/naotoo1/prosemble";
    changelog = "https://github.com/naotoo1/prosemble/releases/tag/v${finalAttrs.version}";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ naotoo1 ];
  };
})

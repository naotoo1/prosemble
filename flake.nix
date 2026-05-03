{
  description = "Prosemble — JAX-based prototype machine learning library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;

        prosemble = python.pkgs.buildPythonPackage {
          pname = "prosemble";
          version = "1.0.0";
          pyproject = true;
          src = ./.;

          build-system = with python.pkgs; [
            setuptools
            setuptools-scm
          ];

          dependencies = with python.pkgs; [
            numpy
            scikit-learn
            matplotlib
          ];

          optional-dependencies = with python.pkgs; {
            jax = [ jax jaxlib chex optax ];
          };

          doCheck = false;

          pythonImportsCheck = [
            "prosemble"
            "prosemble.models"
            "prosemble.core"
          ];

          meta = with pkgs.lib; {
            description = "JAX-based prototype machine learning library";
            homepage = "https://github.com/naotoo1/prosemble";
            license = licenses.mit;
          };
        };
      in
      {
        packages = {
          default = prosemble;
          prosemble = prosemble;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            (python.withPackages (ps: with ps; [
              prosemble
              jax
              jaxlib
              chex
              optax
              pytest
              ipython
            ]))
          ];
        };
      }
    );
}

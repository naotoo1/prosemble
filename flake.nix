{
  description = "Development environment for Prosemble project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: {
      devShells.default = let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonPackages = pkgs.python312Packages;
      in pkgs.mkShell {
        buildInputs = with pkgs; [
          bashInteractive
          git
          git-lfs
          nixpkgs-fmt
          python312
          pythonPackages.numpy
          pythonPackages.matplotlib
          pythonPackages.scikit-learn
          pythonPackages.pandas
          pythonPackages.scipy
          pythonPackages.pip
          pythonPackages.uv
          docker
          docker-compose
        ];
        shellHook = ''
          # Set up environment variables
          export PROJECT_DIR=$PWD
          export PYTHONPATH="${pythonPackages.sitePackages}:$PYTHONPATH"
        '';
      };
    });
}
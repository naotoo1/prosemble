{
  description = "Development environment for Prosemble project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, flake-utils, devenv, ... }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.${system}.default = devenv.lib.mkShell {
          pkgs = pkgs;
          shell = pkgs.bashInteractive;  # Explicitly set shell to bash
          env = {
            # Add any environment variables needed
            ENV_VAR = "value";
          };
        };
      });
}

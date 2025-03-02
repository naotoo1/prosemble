{
  description = "Development environment for Prosemble project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, flake-utils, devenv, ... }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: {
      devShells.default = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          ./devenv.nix
        ];
      };
    });
}
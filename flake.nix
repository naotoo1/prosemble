{
  description = "Development environment for Prosemble project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, flake-utils, devenv, ... }@inputs:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: {
      devShells.default = let
        pkgs = nixpkgs.legacyPackages.${system};
      in devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          ./devenv.nix
        ];
      };
    });
}
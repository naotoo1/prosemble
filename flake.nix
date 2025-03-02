{
  description = "Reproducible Python development environment with devenv and Nix flakes";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # Latest Nixpkgs
    devenv.url = "github:cachix/devenv/latest"; # devenv for environment management
  };

  outputs = { self, nixpkgs, devenv, ... }:
    let
      system = "x86_64-linux"; # Adjust for your system (e.g., "aarch64-linux" for ARM)
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = devenv.lib.mkShell {
        inherit pkgs;
        modules = [ ./devenv.nix ]; # Use your devenv.nix for configuration
      };
    };
}

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
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          bashInteractive
          # Add any other dependencies you need here
        ];
        shellHook = ''
          # Set up environment variables or other settings
          export ENV_VAR="value"
        '';
      };
    });
}

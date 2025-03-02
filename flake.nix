# flake.nix
{
  description = "Development environment for Prosemble project";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, devenv, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system: 
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            {
              # Project configuration
              packages = with pkgs; [
                git
                git-lfs
                nixpkgs-fmt
                docker
                docker-compose
              ];

              languages.python = {
                enable = true;
                package = pkgs.python312;
                version = "3.12";
                uv.enable = true;
                venv.enable = true;
              };

              # Python packages
              packages = with pkgs.python312Packages; [
                numpy
                matplotlib
                scikit-learn
                pandas
                scipy
                pip
              ];
              
              # Environment variables
              env.PYTHONPATH = "${pkgs.python312Packages.python.sitePackages}";
              
              # Shell configuration
              starship.enable = true;
            }
          ];
        };
      }
    );
}
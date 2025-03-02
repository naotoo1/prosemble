{
  description = "Development environment for Prosemble project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    flake-utils.url = "github:numtide/flake-utils";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, flake-utils, ... } @ inputs:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = devenv.lib.mkShell {
        pkgs = pkgs;
        shell = pkgs.bashInteractive;  # Explicitly set shell to bash
        modules = [
          ({ pkgs, ... }: {
            # Project-specific packages
            packages = [
              pkgs.git
              pkgs.git-lfs
              pkgs.docker
              pkgs.docker-compose
              pkgs.nixpkgs-fmt
            ];

            # Python environment setup
            languages.python = {
              enable = true;
              package = pkgs.python3;
              version = "3.12";
              venv.enable = true;
            };

            # Other environment variables
            env.PYTHONPATH = "${pkgs.python3.sitePackages}";
          })
        ];
      };
    });
}

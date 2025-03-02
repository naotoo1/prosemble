{
  description = "Development environment for Prosemble project";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    flake-utils.url = "github:numtide/flake-utils";
  };
  
  outputs = { self, nixpkgs, devenv, flake-utils, ... }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = devenv.lib.mkShell {
        pkgs = pkgs;
        shell = pkgs.bashInteractive;
        modules = [
          {
            # Project configuration
            packages = with pkgs; [
              git
              git-lfs
              nixpkgs-fmt
              docker
              docker-compose
              
              # Python packages
              (python312.withPackages (ps: with ps; [
                numpy
                matplotlib
                scikit-learn
                pandas
                scipy
                pip
              ]))
            ];

            languages.python = {
              enable = true;
              package = pkgs.python312;
              version = "3.12";
              uv.enable = true;
              venv.enable = true;
            };
            
            # Environment variables
            env.PYTHONPATH = "${pkgs.python312.sitePackages}";
            
            # Shell configuration
            starship.enable = true;
          }
        ];
      };
    });
}

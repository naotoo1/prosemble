# one with the docker container
{ pkgs, lib, config, inputs, ... }:

let
  pythonPackages = pkgs.python312Packages;
  project_dir = "${config.env.DEVENV_ROOT}";
  env.PYTHONPATH = "${pkgs.python312Packages.sitePackages}";
in {
  cachix.enable = false;
  env.PROJECT_DIR = project_dir;

  packages = with pkgs; [
    git
    git-lfs
    nixpkgs-fmt
    pythonPackages.numpy
    pythonPackages.matplotlib
    pythonPackages.scikit-learn
    pythonPackages.pandas
    pythonPackages.scipy
    pythonPackages.pip
    docker
    docker-compose

  ];

  # Add Docker support
  # containers.docker = {
  #   isSystemd = true;
  #   ephemeral = true;
  #   privateUsers = true;
  #   extraOptions = ["--network=host"];
  # };

  languages = {
    python = {
      enable = true;
      package = pythonPackages.python;
      uv.enable = true;
      venv = {
        enable = true;
      };
    };
  };

  starship.enable = true;
}


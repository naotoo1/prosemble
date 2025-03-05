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
  ];


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
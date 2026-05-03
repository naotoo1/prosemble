{ pkgs, lib, config, inputs, ... }:

{
  cachix.enable = false;

  env = {
    LD_LIBRARY_PATH = lib.concatStringsSep ":" [
      (lib.makeLibraryPath (with pkgs; [ stdenv.cc.cc.lib zlib ]))
      "/run/opengl-driver/lib"
    ];
  };

  packages = with pkgs; [
    git
    git-lfs
    gh
    zlib
  ];

  languages = {
    python = {
      enable = true;
      package = pkgs.python312;
      uv = {
        enable = true;
        sync.enable = true;
      };
      venv = {
        enable = true;
      };
    };
  };

  starship.enable = true;

  enterShell = ''
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    echo "prosemble (Python 3.12 + JAX) — run 'devenv info' for available scripts"
  '';
}

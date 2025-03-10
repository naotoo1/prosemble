{ pkgs, lib, config, inputs, ... }:
let
  pythonPackages = pkgs.python312Packages;
  project_dir = "${config.env.DEVENV_ROOT}";
in {
  cachix.enable = false;
  
  env = {
    PROJECT_DIR = project_dir;

    # Ensure Nix packages are discoverable by Python
    PYTHONPATH = lib.makeSearchPath "lib/python3.12/site-packages" [
      pythonPackages.numpy
      pythonPackages.matplotlib
      pythonPackages.scikit-learn
      pythonPackages.pandas
      pythonPackages.scipy
    ];
  };

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
    nix-prefetch-git
  ];

  languages = {
    python = {
      enable = true;
      package = pythonPackages.python;
      uv = {
        enable = true;
      };
      venv = {
        enable = true;
      };
    };
  };

  scripts = {
    # Generates requirements.in without Nix packages
    generate-requirements.exec = ''
      echo "Generating requirements.in without Nix packages..."
      cat > requirements.in << EOL
# Non-Nix packages
-e .
EOL
      echo "requirements.in generated."
    '';

    # Updates uv.lock using requirements.in - ensures old lock file is replaced
    update-lock-files.exec = ''
      echo "Updating lock files..."
      
      # Generate requirements.in
      $PROJECT_DIR/generate-requirements

      # Remove existing uv.lock if it exists
      if [ -f uv.lock ]; then
        echo "Removing existing uv.lock file..."
        rm uv.lock
      fi

      # Generate new uv.lock
      echo "Generating new uv.lock file..."
      uv pip compile requirements.in -o uv.lock

      echo "Lock files updated successfully."
    '';

    # Installs non-Nix Python packages from uv.lock
    install-from-lock.exec = ''
      echo "Installing Python packages from uv.lock..."
      if [ -f uv.lock ]; then
        uv pip sync uv.lock
        echo "Non-Nix packages installed successfully."
      else
        echo "uv.lock not found. Run update-lock-files first."
      fi
    '';

    # Handles the entire flow from requirements to installation
    setup-python-env.exec = ''
      echo "=== Setting up Python environment ==="
      echo ""
      
      # Step 1: Generate requirements.in
      echo "STEP 1: Generating requirements.in..."
      $PROJECT_DIR/generate-requirements
      
      # Step 2: Update lock files
      echo ""
      echo "STEP 2: Updating lock files..."
      
      # Remove existing uv.lock if it exists
      if [ -f uv.lock ]; then
        echo "Removing existing uv.lock file..."
        rm uv.lock
      fi
      
      # Generate new uv.lock
      echo "Generating new uv.lock file..."
      uv pip compile requirements.in -o uv.lock
      
      # Step 3: Install packages from lock
      echo ""
      echo "STEP 3: Installing packages from lock..."
      uv pip sync uv.lock
      
      echo ""
      echo "=== Python environment setup complete ==="
      echo "Local package installed in editable mode."
    '';

    # Use install-from-lock instead of duplicating package installation
    quick-install-non-nix-packages.exec = ''
      echo "Quick installing non-Nix packages..."
      # Check if requirements.in exists, if not generate it
      if [ ! -f requirements.in ]; then
        $PROJECT_DIR/generate-requirements
      fi
      
      # Check if uv.lock exists, if not generate it
      if [ ! -f uv.lock ]; then
        echo "Generating uv.lock file..."
        uv pip compile requirements.in -o uv.lock
      fi
      
      # Install from the lock file
      uv pip sync uv.lock
      echo "Non-Nix packages installed successfully."
    '';

    # CPU-only container for systems without GPUs
    create-cpu-container.exec = ''
      echo "Creating CPU-only container from lock files..."
      
      # Ensure uv.lock exists
      if [ ! -f uv.lock ]; then
        echo "uv.lock not found. Generating requirements.in and uv.lock now..."
        $PROJECT_DIR/generate-requirements
        uv pip compile requirements.in -o uv.lock
      fi
      
      # Use the same approach as GPU container - create requirements without local package
      if [ ! -f requirements-docker.in ]; then
        grep -v "^-e \." requirements.in > requirements-docker.in
      fi
      
      # Create frozen requirements from the Docker-specific input
      uv pip compile requirements-docker.in -o requirements-frozen-docker.txt
      
      # Create CPU Dockerfile
      cat > Dockerfile.cpu << EOF
      FROM python:3.12-slim
      
      # System dependencies
      RUN apt-get update && apt-get install -y \\
          git \\
          git-lfs \\
          && rm -rf /var/lib/apt/lists/*
      
      # Create working directory
      WORKDIR /app
      
      # Upgrade pip and setuptools first
      RUN pip3 install --no-cache-dir --upgrade pip setuptools>=61
      
      # Copy the entire project
      COPY . /app
      
      # Copy Docker-specific frozen requirements
      COPY requirements-frozen-docker.txt /app/requirements-frozen.txt
      
      # Install dependencies from frozen requirements
      RUN pip3 install --no-cache-dir -r requirements-frozen.txt
      
      # Install the local package in editable mode for development
      RUN pip3 install --no-cache-dir -e .
      
      # Default command
      CMD ["python3"]
      EOF
      
      echo "Building CPU Docker image from frozen requirements..."
      docker build -t cpu-env-locked -f Dockerfile.cpu .
      echo "CPU image built successfully."
    '';

    run-cpu-container.exec = ''
      echo "Running CPU-only container..."
      docker run -it --rm \
        -v "$(pwd):/app" \
        cpu-env-locked bash
    '';   

  };

  starship.enable = true;
  
  enterShell = ''
    echo "=== Python Environment Setup ==="
    echo ""
    echo "Lock file commands:"
    echo "- Generate requirements.in:  run 'generate-requirements'"
    echo "- Update lock files:         run 'update-lock-files'"
    echo "- Install from lock:         run 'install-from-lock'"
    echo "- Complete setup (all steps): run 'setup-python-env'"
    echo ""
    echo "Docker commands:"
    echo "- Create CPU container:      run 'create-cpu-container'"
    echo "- Run CPU container:         run 'run-cpu-container'"
  '';
}
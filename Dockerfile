FROM ubuntu:22.04

# Install minimal required packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    xz-utils \
    procps \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Nix
RUN curl -L https://nixos.org/nix/install | sh -s -- --daemon --yes \
    && echo "trusted-users = root" >> /etc/nix/nix.conf

# Set up environment variables
ENV PATH="/nix/var/nix/profiles/default/bin:$PATH"
ENV NIX_PATH="nixpkgs=channel:nixos-unstable"

# Install direnv
RUN curl -sfL https://direnv.net/install.sh | bash

# Set up shell for direnv
RUN echo 'eval "$(direnv hook bash)"' >> /root/.bashrc

# Install devenv
RUN . /root/.nix-profile/etc/profile.d/nix.sh && \
    nix-env -if https://github.com/cachix/devenv/tarball/latest

# Create app directory
WORKDIR /app

# Set up direnv
RUN echo 'use flake' > .envrc 

# Create a wrapper script to load the devenv environment
RUN echo '#!/bin/bash\n\
source /root/.nix-profile/etc/profile.d/nix.sh\n\
eval "$(devenv shell --print-bash)"\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Install essential Python tools globally to help with bootstrapping
RUN . /root/.nix-profile/etc/profile.d/nix.sh && \
    nix-env -i python3 && \
    nix-env -i uv

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash", "-l"]
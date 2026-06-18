#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "User: $(whoami)"
echo "Current directory: $(pwd)"

if [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
else
    WORKSPACE_DIR="$HOME/workspace"
    mkdir -p "$WORKSPACE_DIR"
fi

echo "Workspace directory: $WORKSPACE_DIR"

install_apt_packages() {
    echo "Installing system packages..."

    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y \
            build-essential \
            git \
            wget \
            unzip \
            tmux \
            xvfb \
            libgl1 \
            libglib2.0-0 \
            libxrender1 \
            libxext6 \
            libsm6 \
            libx11-6 \
            libegl1 \
            libopengl0 \
            mesa-utils \
            python3-venv \
            python3-pip
    else
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            build-essential \
            git \
            wget \
            unzip \
            tmux \
            xvfb \
            libgl1 \
            libglib2.0-0 \
            libxrender1 \
            libxext6 \
            libsm6 \
            libx11-6 \
            libegl1 \
            libopengl0 \
            mesa-utils \
            python3-venv \
            python3-pip
    fi
}

setup_python_env() {
    echo "Creating Python virtual environment..."

    cd "$PROJECT_ROOT"

    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi

    source .venv/bin/activate

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

    echo "Python environment ready."
    python --version
}

install_holoocean_client() {
    echo "Installing HoloOcean Python client from GitHub..."

    mkdir -p "$WORKSPACE_DIR/external"

    echo "Checking GitHub SSH access..."
    ssh -T git@github.com || true

    if [ ! -d "$WORKSPACE_DIR/external/holoocean" ]; then
        echo "Cloning HoloOcean repository using SSH..."

        git clone git@github.com:byu-holoocean/HoloOcean.git "$WORKSPACE_DIR/external/holoocean"    
    else
        echo "HoloOcean repo already exists. Pulling latest changes..."
        cd "$WORKSPACE_DIR/external/holoocean"
        git pull || true
    fi

    cd "$WORKSPACE_DIR/external/holoocean/client"

    source "$PROJECT_ROOT/.venv/bin/activate"
    python -m pip install .

    cd "$PROJECT_ROOT"
}

main() {
    install_apt_packages
    setup_python_env
    install_holoocean_client

    source "$PROJECT_ROOT/.venv/bin/activate"

    echo "Testing GPU..."
    python scripts/test_gpu.py

    echo "Installing Ocean package..."
    python scripts/install_ocean_package.py

    echo "Checking HoloOcean installation..."
    python scripts/check_holoocean_install.py

    echo "Phase 1 setup complete."
}

main
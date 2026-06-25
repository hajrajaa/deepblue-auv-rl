#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "User: $(whoami)"
echo "Home directory: $HOME"
echo "Current directory: $(pwd)"

if [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
else
    WORKSPACE_DIR="$HOME/workspace"
    mkdir -p "$WORKSPACE_DIR"
fi

echo "Workspace directory: $WORKSPACE_DIR"


run_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "WARNING: Cannot run root command because sudo is not installed."
        echo "Run this script as root, or install apt packages manually first."
        return 1
    fi
}

install_apt_packages() {
    echo "Installing system packages..."

    if ! run_root apt-get update; then
        echo "Skipping apt install because we are not root and sudo is missing."
        echo "This is OK only if you already installed the system packages."
        return 0
    fi

    run_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        git \
        openssh-client \
        ca-certificates \
        curl \
        wget \
        unzip \
        tmux \
        xvfb \
        python3-venv \
        python3-pip \
        vulkan-tools \
        libvulkan1 \
        mesa-vulkan-drivers \
        mesa-utils \
        libgl1 \
        libegl1 \
        libglvnd0 \
        libopengl0 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-dri3-0 \
        libxcb-present0 \
        libxkbcommon0

    echo "System packages installed."
}

setup_holoocean_persistent_dir() {
    echo "Setting persistent HoloOcean package directory..."

    mkdir -p "$WORKSPACE_DIR/holoocean_data"
    mkdir -p "$HOME/.local/share"

    if [ -e "$HOME/.local/share/holoocean" ] && [ ! -L "$HOME/.local/share/holoocean" ]; then
        BACKUP="$HOME/.local/share/holoocean_backup_$(date +%Y%m%d_%H%M%S)"
        echo "Existing HoloOcean folder found. Moving it to: $BACKUP"
        mv "$HOME/.local/share/holoocean" "$BACKUP"

        if command -v rsync >/dev/null 2>&1; then
            rsync -a "$BACKUP/" "$WORKSPACE_DIR/holoocean_data/" || true
        else
            cp -a "$BACKUP/." "$WORKSPACE_DIR/holoocean_data/" || true
        fi
    fi

    if [ ! -L "$HOME/.local/share/holoocean" ]; then
        ln -s "$WORKSPACE_DIR/holoocean_data" "$HOME/.local/share/holoocean"
    fi

    echo "HoloOcean data path:"
    ls -la "$HOME/.local/share/holoocean"
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

check_network_for_holoocean() {
    echo "Checking HoloOcean backend URL..."

    source "$PROJECT_ROOT/.venv/bin/activate"

    python - <<'PY' || true
import holoocean.packagemanager as pm
print("HoloOcean backend:", getattr(pm, "BACKEND_URL", "UNKNOWN"))
PY
}
install_ocean_package_safe() {
    echo "Installing Ocean package..."

    source "$PROJECT_ROOT/.venv/bin/activate"

    set +e
    python scripts/install_ocean_package.py
    STATUS=$?
    set -e

    if [ "$STATUS" -ne 0 ]; then
        echo ""
        echo "WARNING: Ocean package installation failed."
        echo "This is usually because the HoloOcean backend is unreachable:"
        echo "  [Errno 113] No route to host"
        echo ""
        echo "The Python client may still be installed correctly."
        echo "But training cannot run until Ocean appears in installed packages."
        echo ""
        return 0
    fi
}

main() {
    install_apt_packages
    setup_holoocean_persistent_dir
    setup_python_env
    install_holoocean_client
    check_network_for_holoocean

    source "$PROJECT_ROOT/.venv/bin/activate"

    echo "Testing GPU..."
    python scripts/test_gpu.py || true

    install_ocean_package_safe

    echo "Checking HoloOcean installation..."
    python scripts/check_holoocean_install.py || true

    echo "Setup script finished."
    echo ""
    echo "Important:"
    echo "If Installed packages is still [], Ocean is not installed."
    echo "That is a HoloOcean backend/network problem, not a Python import problem."
}

main
#!/bin/bash
set -e  # Exit immediately if any command fails

echo "========================================================"
echo "  Starting TorchCodec Installation for NVIDIA DGX (ARM) "
echo "========================================================"

# 1. Install System Dependencies (Requires Sudo)
# These are needed to link against FFmpeg and compile C++ extensions.
echo "[1/5] Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
    pkg-config \
    cmake \
    build-essential

# 2. Install Python Build Tools
# We need these in the current env because we are disabling build isolation.
echo "[2/5] Installing Python build tools (pybind11, cmake, numpy)..."
uv pip install pybind11 numpy cmake

# 3. Configure CMake Environment
# Explicitly point CMake to the pybind11 installation directory.
echo "[3/5] Configuring build environment..."
export pybind11_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "      - pybind11 found at: $pybind11_DIR"

# 4. Install TorchCodec
# - I_CONFIRM...=1: Bypasses FFmpeg license prompt
# - --no-build-isolation: Uses system torch/ffmpeg (Critical for DGX)
echo "[4/5] Building and installing TorchCodec from source..."
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 uv pip install \
    --no-build-isolation \
    "torchcodec @ git+https://github.com/pytorch/torchcodec.git@v0.9.1"

# 5. Verify Installation
echo "[5/5] Verifying installation..."
uv run python -c "import torchcodec; print(f'SUCCESS: TorchCodec v{torchcodec.__version__} installed successfully.')"

echo "========================================================"
echo "  Installation Complete!"
echo "========================================================"
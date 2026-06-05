#!/usr/bin/env bash
# Install system prerequisites for quantfx-compiler (Ubuntu/Debian).
set -euo pipefail

echo "==> Installing LLVM 18 + MLIR tools..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  llvm-18 llvm-18-dev llvm-18-tools \
  mlir-18-tools libmlir-18-dev \
  ccache build-essential

echo "==> Creating Python virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source .venv/bin/activate"
echo ""
echo "Build the compiler (once source is added):"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DMLIR_DIR=\$(llvm-config-18 --cmakedir)/../mlir -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"

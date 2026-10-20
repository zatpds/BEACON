#!/usr/bin/env bash
#
# One-shot environment setup for BEACON.
#
# Creates a conda environment, installs PyTorch (GPU build preferred, CPU
# fallback) and the Python dependencies, and installs the three vendored
# simulator packages (robomimic, robosuite, mimicgen) from packages/ in
# editable mode.
#
# Prerequisites: Linux, NVIDIA GPU (optional; CPU works but is impractically
# slow for image-based diffusion policies).
# Edit CONDA_DIR below if your conda lives elsewhere.

set -euo pipefail

ENV_NAME="beacon"
PY_VER="3.9"
CONDA_DIR="${HOME}/miniconda3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

echo "[1/6] Installing system dependencies (apt)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential git wget curl cmake ca-certificates \
  libgl1 libglx-mesa0 libglib2.0-0 \
  libosmesa6-dev libglfw3-dev patchelf

echo "[2/6] Ensuring Miniconda exists..."
if [ ! -d "${CONDA_DIR}" ]; then
  mkdir -p "$(dirname "${CONDA_DIR}")"
  TMP_SH="/tmp/miniconda.sh"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${TMP_SH}"
  bash "${TMP_SH}" -b -p "${CONDA_DIR}"
  rm -f "${TMP_SH}"
fi
source "${CONDA_DIR}/etc/profile.d/conda.sh"

echo "[3/6] Creating conda environment '${ENV_NAME}' (Python ${PY_VER})..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  - Env already exists, skipping create."
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
conda activate "${ENV_NAME}"

echo "[4/6] Installing PyTorch (prefers GPU build; falls back to CPU)..."
# Experiments were run with torch 2.8.0+cu128; any torch >= 2.0 should work.
TORCH_VER="2.8.0"
set +e
pip install "torch==${TORCH_VER}" torchvision --index-url https://download.pytorch.org/whl/cu128
TORCH_OK=$?
set -e
if [ "${TORCH_OK}" -ne 0 ]; then
  echo "  - CUDA 12.8 wheel unavailable; falling back to CPU build..."
  pip install "torch==${TORCH_VER}" torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[5/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${REPO_DIR}/requirements.txt"

echo "[6/6] Installing vendored packages (robomimic, robosuite, mimicgen)..."
pip install -r "${REPO_DIR}/packages/robosuite/requirements.txt"
pip install -e "${REPO_DIR}/packages/robosuite"
pip install -e "${REPO_DIR}/packages/mimicgen"
pip install -e "${REPO_DIR}/packages/robomimic"

echo
echo "Done. To use the environment in a new shell:"
echo "  source \"${CONDA_DIR}/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "  export MUJOCO_GL=osmesa   # or egl / glfw"

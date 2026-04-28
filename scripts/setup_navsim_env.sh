#!/usr/bin/env bash
set -euo pipefail

# Install the tested FastWAM-NavSIM Python environment inside the currently
# active conda environment. Create and activate the conda env before running:
#   conda create -n fastwam python=3.10 -y
#   conda activate fastwam
#   bash scripts/setup_navsim_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/third_party/navsim}"
NUPLAN_DEVKIT_PACKAGE="${NUPLAN_DEVKIT_PACKAGE:-${REPO_ROOT}/third_party/nuplan-devkit-v1.2.tar.gz}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${NAVSIM_DEVKIT_ROOT}" ]]; then
  echo "ERROR: NAVSIM_DEVKIT_ROOT does not exist: ${NAVSIM_DEVKIT_ROOT}" >&2
  echo "Set NAVSIM_DEVKIT_ROOT or keep the vendored devkit at third_party/navsim." >&2
  exit 1
fi

if [[ ! -f "${NUPLAN_DEVKIT_PACKAGE}" && ! -d "${NUPLAN_DEVKIT_PACKAGE}" ]]; then
  echo "ERROR: NUPLAN_DEVKIT_PACKAGE does not exist: ${NUPLAN_DEVKIT_PACKAGE}" >&2
  echo "Set NUPLAN_DEVKIT_PACKAGE or keep the vendored package at third_party/nuplan-devkit-v1.2.tar.gz." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -U pip wheel

# Install nuPlan from the vendored package first so NAVSIM can import it without
# needing to clone from GitHub on restricted machines.
"${PYTHON_BIN}" -m pip install "${NUPLAN_DEVKIT_PACKAGE}"

# Install NAVSIM from vendored source. Use --no-deps because the FastWAM-tested
# dependency versions are restored/installed below.
"${PYTHON_BIN}" -m pip install -e "${NAVSIM_DEVKIT_ROOT}" --no-deps

# Install the NAVSIM-side runtime packages observed in the working environment.
"${PYTHON_BIN}" -m pip install \
  opencv-python==4.9.0.80 \
  scikit-learn==1.2.2 \
  positional-encodings==6.0.1 \
  geopandas==1.1.3 \
  shapely==2.1.2 \
  ray==2.55.1 \
  pytorch-lightning==2.2.1 \
  tensorboard==2.16.2 \
  protobuf==4.25.3 \
  timm==1.0.26

# Restore FastWAM-tested CUDA/PyTorch versions before installing this package.
"${PYTHON_BIN}" -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url "${CUDA_INDEX_URL}"

# Install FastWAM-NavSIM and its pinned dependencies from pyproject.toml.
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}" --extra-index-url "${CUDA_INDEX_URL}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as im
import cv2
import hydra
import navsim
import nuplan
import numpy
import torch
import torchvision

print("Environment check:")
print("  torch", torch.__version__)
print("  torchvision", torchvision.__version__)
print("  numpy", numpy.__version__)
print("  hydra", hydra.__version__)
print("  opencv", cv2.__version__)
print("  navsim", im.version("navsim"), navsim.__file__)
print("  nuplan-devkit", im.version("nuplan-devkit"), nuplan.__file__)
PY

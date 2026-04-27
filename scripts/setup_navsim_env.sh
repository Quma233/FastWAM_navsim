#!/usr/bin/env bash
set -euo pipefail

# Install the tested FastWAM-NavSIM Python environment inside the currently
# active conda environment. Create and activate the conda env before running:
#   conda create -n fastwam python=3.10 -y
#   conda activate fastwam
#   export NAVSIM_DEVKIT_ROOT=/path/to/navsim_dataset/navsim
#   bash scripts/setup_navsim_env.sh

if [[ -z "${NAVSIM_DEVKIT_ROOT:-}" ]]; then
  echo "ERROR: NAVSIM_DEVKIT_ROOT is not set." >&2
  echo "Set it to the NAVSIM devkit root, e.g. /path/to/navsim_dataset/navsim" >&2
  exit 1
fi

if [[ ! -d "${NAVSIM_DEVKIT_ROOT}" ]]; then
  echo "ERROR: NAVSIM_DEVKIT_ROOT does not exist: ${NAVSIM_DEVKIT_ROOT}" >&2
  exit 1
fi

CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install -U pip wheel

# Install NAVSIM/nuPlan first. NAVSIM's requirements may temporarily install
# older torch/numpy/hydra versions; the FastWAM install below restores the
# versions tested for this repository.
"${PYTHON_BIN}" -m pip install -e "${NAVSIM_DEVKIT_ROOT}"

# Restore FastWAM-tested CUDA/PyTorch versions before installing this package.
"${PYTHON_BIN}" -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url "${CUDA_INDEX_URL}"

# Install FastWAM-NavSIM and its pinned dependencies from pyproject.toml.
"${PYTHON_BIN}" -m pip install -e . --extra-index-url "${CUDA_INDEX_URL}"

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

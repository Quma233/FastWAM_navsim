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
ENV_LOCK_FILE="${ENV_LOCK_FILE:-${REPO_ROOT}/requirements/fastwam_navsim_env.txt}"
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

if [[ ! -f "${ENV_LOCK_FILE}" ]]; then
  echo "ERROR: ENV_LOCK_FILE does not exist: ${ENV_LOCK_FILE}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -U pip wheel

# Install the vendored devkits themselves first. Dependencies are installed from
# the tested lock file below, not from NAVSIM's conflicting requirements.txt.
"${PYTHON_BIN}" -m pip install "${NUPLAN_DEVKIT_PACKAGE}" --no-deps
"${PYTHON_BIN}" -m pip install -e "${NAVSIM_DEVKIT_ROOT}" --no-deps

# Reproduce the tested fastwam conda environment. The lock file was exported
# from the working container and excludes only local editable packages and local
# file URLs for FastWAM, NAVSIM, and nuPlan.
"${PYTHON_BIN}" -m pip install \
  -r "${ENV_LOCK_FILE}" \
  --extra-index-url "${CUDA_INDEX_URL}"

# Install this repo last without dependency resolution. All dependencies should
# already match the lock file, so this avoids pip upgrading torch to newer CUDA
# wheels such as 2.11.
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}" --no-deps

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as im
import cv2
import hydra
import navsim
import nuplan
import numpy
import torch
import torchvision

expected = {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "numpy": "1.26.4",
    "hydra-core": "1.3.2",
    "accelerate": "1.12.0",
    "deepspeed": "0.18.5",
    "navsim": "1.1.0",
    "nuplan-devkit": "1.2.0",
}
actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "hydra-core": hydra.__version__,
    "accelerate": im.version("accelerate"),
    "deepspeed": im.version("deepspeed"),
    "navsim": im.version("navsim"),
    "nuplan-devkit": im.version("nuplan-devkit"),
}

print("Environment check:")
for key, value in actual.items():
    print(f"  {key} {value}")
print("  opencv", cv2.__version__)
print("  navsim_path", navsim.__file__)
print("  nuplan_path", nuplan.__file__)

bad = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
if bad:
    details = ", ".join(f"{key}: got {got}, expected {want}" for key, (got, want) in bad.items())
    raise SystemExit(f"Environment version check failed: {details}")
PY

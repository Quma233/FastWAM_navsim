#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OBS_PERSONAL_ROOT="${OBS_PERSONAL_ROOT:-obs://yw-2030-gy/external/personal/f50000365}"
OBS_REPO_ROOT="${OBS_REPO_ROOT:-${OBS_PERSONAL_ROOT}/FastWAM_navsim_di}"
OBS_DATA_ROOT="${OBS_DATA_ROOT:-${OBS_PERSONAL_ROOT}/navsim_dataset}"
OBS_CHECKPOINTS_ROOT="${OBS_CHECKPOINTS_ROOT:-${OBS_REPO_ROOT}/checkpoints}"
OBS_TEXT_EMBEDS_ROOT="${OBS_TEXT_EMBEDS_ROOT:-${OBS_REPO_ROOT}/data/text_embeds_cache}"
OBS_RESULTS_ROOT="${OBS_RESULTS_ROOT:-${OBS_REPO_ROOT}/runs}"
MOXING_WHEEL_OBS_URI="${MOXING_WHEEL_OBS_URI:-obs://yw-ads-training-gy1/data/external/personal/z00009214/moxing_framework-2.5.0rc6-py2.py3-none-any.whl}"
MOXING_WHEEL_NAME="${MOXING_WHEEL_NAME:-moxing_framework-2.5.0rc6-py2.py3-none-any.whl}"

WORKSPACE="${WORKSPACE:-/home/ma-user/code}"
LOCAL_REPO_ROOT="${LOCAL_REPO_ROOT:-${WORKSPACE}/FastWAM_navsim_di}"
BOOTSTRAP_CONDA_ENV="${BOOTSTRAP_CONDA_ENV:-qwen3}"
BOOTSTRAP_CONDA_BIN="${BOOTSTRAP_CONDA_BIN:-/root/miniconda3/bin/conda}"
BOOTSTRAP_CONDA_SH="${BOOTSTRAP_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
BOOTSTRAP_MOXING_WHEEL_PATH="${BOOTSTRAP_MOXING_WHEEL_PATH:-/tmp/${MOXING_WHEEL_NAME}}"

export S3_ENDPOINT="${S3_ENDPOINT:-https://obs.cn-southwest-2.myhuaweicloud.com}"
export S3_USE_HTTPS="${S3_USE_HTTPS:-0}"
export ACCESS_KEY_ID="${ACCESS_KEY_ID:-HPUACJ8EWHNONWBP1PGQ}"
export SECRET_ACCESS_KEY="${SECRET_ACCESS_KEY:-rx3ITEWElWS8dZnPHmFMmMiiGkQ2pk5FguQ0d1QN}"

enabled() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }

moxing_ok() {
  python - <<'PY' >/dev/null 2>&1
import moxing as mox
if not hasattr(mox, "file"):
    raise SystemExit(1)
PY
}

activate_bootstrap_env() {
  [[ -n "${BOOTSTRAP_CONDA_ENV}" ]] || return 0
  local hook=""
  if [[ -x "${BOOTSTRAP_CONDA_BIN}" ]]; then
    hook="$(${BOOTSTRAP_CONDA_BIN} shell.bash hook 2>/dev/null || true)"
  elif command -v conda >/dev/null 2>&1; then
    hook="$(conda shell.bash hook 2>/dev/null || true)"
  fi
  if [[ -n "${hook}" ]]; then
    eval "${hook}"
  elif [[ -f "${BOOTSTRAP_CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    . "${BOOTSTRAP_CONDA_SH}"
  elif [[ -x "${BOOTSTRAP_CONDA_BIN}" ]]; then
    export PATH="$(dirname "${BOOTSTRAP_CONDA_BIN}"):${PATH}"
  elif ! command -v conda >/dev/null 2>&1; then
    echo "[bootstrap] conda not found; cannot activate ${BOOTSTRAP_CONDA_ENV}" >&2
    exit 1
  fi
  echo "[bootstrap] activating conda env: ${BOOTSTRAP_CONDA_ENV}"
  conda activate "${BOOTSTRAP_CONDA_ENV}"
  export FASTWAM_BOOTSTRAP_CONDA_ACTIVE=1
}

install_moxing_wheel_from_obs() {
  local wheel_path="$1" label="$2"
  moxing_ok || { echo "${label} moxing.file is required before downloading ${MOXING_WHEEL_NAME}" >&2; exit 1; }
  echo "${label} downloading MoXing wheel: ${MOXING_WHEEL_OBS_URI} -> ${wheel_path}"
  python - "${MOXING_WHEEL_OBS_URI}" "${wheel_path}" <<'PY'
import sys
import moxing
moxing.file.copy(sys.argv[1], sys.argv[2])
PY
  [[ -f "${wheel_path}" ]] || { echo "${label} missing wheel: ${wheel_path}" >&2; exit 1; }
  echo "${label} installing MoXing wheel: ${wheel_path}"
  python -m pip install "${wheel_path}" --upgrade-strategy only-if-needed
  moxing_ok || { echo "${label} moxing.file unavailable after wheel install" >&2; exit 1; }
}

sync_repo_from_obs() {
  local src="${OBS_REPO_ROOT%/}/" dst="${LOCAL_REPO_ROOT%/}/"
  mkdir -p "$(dirname "${LOCAL_REPO_ROOT}")"
  echo "[bootstrap] syncing repo: ${src} -> ${dst}"
  python - "${src}" "${dst}" <<'PY'
import sys
import traceback
import moxing as mox
try:
    mox.file.copy_parallel(sys.argv[1], sys.argv[2])
except Exception as exc:
    print(f"[bootstrap] repo copy failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()
    raise SystemExit(1)
PY
  [[ -f "${LOCAL_REPO_ROOT}/scripts/run_di_navsim_obs.sh" ]] || {
    echo "[bootstrap] missing synced script: ${LOCAL_REPO_ROOT}/scripts/run_di_navsim_obs.sh" >&2
    exit 1
  }
}

bootstrap_sync_repo_if_needed() {
  [[ "${SYNC_REPO_FROM_OBS:-1}" == "0" || "${FASTWAM_DI_REPO_SYNCED:-0}" == "1" ]] && return 0
  mkdir -p "${LOCAL_REPO_ROOT}"
  [[ "$(cd "${SCRIPT_REPO_ROOT}" && pwd -P)" == "$(cd "${LOCAL_REPO_ROOT}" && pwd -P)" ]] && return 0

  activate_bootstrap_env
  if enabled "${BOOTSTRAP_INSTALL_MOXING_FROM_OBS:-1}"; then
    install_moxing_wheel_from_obs "${BOOTSTRAP_MOXING_WHEEL_PATH}" "[bootstrap]"
  fi
  sync_repo_from_obs
  if [[ "${FASTWAM_BOOTSTRAP_CONDA_ACTIVE:-0}" == "1" ]] && command -v conda >/dev/null 2>&1; then
    echo "[bootstrap] deactivating conda env: ${BOOTSTRAP_CONDA_ENV}"
    conda deactivate || true
  fi
  export FASTWAM_DI_REPO_SYNCED=1
  exec bash "${LOCAL_REPO_ROOT}/scripts/run_di_navsim_obs.sh" "$@"
}
bootstrap_sync_repo_if_needed "$@"

REPO_ROOT="${SCRIPT_REPO_ROOT}"
[[ "${FASTWAM_DI_REPO_SYNCED:-0}" == "1" ]] && REPO_ROOT="${LOCAL_REPO_ROOT}"
cd "${REPO_ROOT}"

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${REPO_ROOT}/conda_envs/fastwam_moxing_di}"
TASK="${TASK:-navsim_v1_uncond_camf0_352x640_1e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:-di_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
SMOKE="${SMOKE:-0}"

if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then NPROC_PER_NODE="$1"; shift; fi
[[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || { echo "ERROR: NPROC_PER_NODE must be an integer, got: ${NPROC_PER_NODE}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RUN_ID HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/third_party/navsim}"
NUPLAN_DEVKIT_PACKAGE="${NUPLAN_DEVKIT_PACKAGE:-${REPO_ROOT}/third_party/nuplan-devkit-v1.2.tar.gz}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/cache/navsim_dataset/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/cache/navsim_dataset/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"

RUN_OUTPUT_DIR="${REPO_ROOT}/runs/${TASK}/${RUN_ID}"
OBS_RUN_OUTPUT_DIR="${OBS_RESULTS_ROOT%/}/${TASK}/${RUN_ID}"
DI_LOG_FILE="${RUN_OUTPUT_DIR}/di_run.log"
mkdir -p "${RUN_OUTPUT_DIR}"
exec > >(tee -a "${DI_LOG_FILE}") 2>&1

log() { printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

mox_copy_parallel() {
  python - "$1" "$2" "${3:-1}" <<'PY'
import sys
import traceback
import moxing as mox
src, dst, required = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try:
    print(f"copy_parallel: {src} -> {dst}")
    mox.file.copy_parallel(src, dst)
except Exception as exc:
    print(f"copy_parallel failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise SystemExit(1 if required else 0)
PY
}

mox_path_exists() {
  python - "$1" <<'PY'
import sys
import moxing as mox
path = sys.argv[1]
try:
    exists = getattr(mox.file, "exists", None)
    if exists is not None and exists(path):
        raise SystemExit(0)
except Exception:
    pass
try:
    mox.file.list_directory(path)
except Exception:
    raise SystemExit(1)
PY
}

mox_list_directory() {
  python - "$1" <<'PY'
import sys
import moxing as mox
path = sys.argv[1]
entries = mox.file.list_directory(path)
print(f"{path}: {len(entries)} entries")
for entry in entries[:5]:
    print(f"  {entry}")
PY
}

upload_results() {
  local exit_code=$?
  set +e
  if enabled "${UPLOAD_RESULTS:-1}" && [[ -d "${RUN_OUTPUT_DIR}" ]]; then
    if moxing_ok; then
      log "Uploading run outputs to ${OBS_RUN_OUTPUT_DIR}/"
      mox_copy_parallel "${RUN_OUTPUT_DIR%/}/" "${OBS_RUN_OUTPUT_DIR}/" 0
    else
      log "Skip result upload because moxing is unavailable."
    fi
  fi
  exit "${exit_code}"
}
trap upload_results EXIT

print_runtime_info() {
  log "repo_root=${REPO_ROOT}"
  log "conda_env_prefix=${CONDA_ENV_PREFIX}"
  log "obs_repo_root=${OBS_REPO_ROOT}"
  log "obs_data_root=${OBS_DATA_ROOT}"
  log "obs_results_root=${OBS_RESULTS_ROOT}"
  log "moxing_wheel_obs_uri=${MOXING_WHEEL_OBS_URI}"
  log "task=${TASK} run_id=${RUN_ID} nproc_per_node=${NPROC_PER_NODE} smoke=${SMOKE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  hostname || true
  pwd
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
}

prepare_conda_env() {
  [[ -d "${CONDA_ENV_PREFIX}" ]] || die "unpacked conda environment directory not found: ${CONDA_ENV_PREFIX}"
  [[ -x "${CONDA_ENV_PREFIX}/bin/python" ]] || die "python not found: ${CONDA_ENV_PREFIX}/bin/python"
  [[ -f "${CONDA_ENV_PREFIX}/bin/activate" ]] || die "activate script not found: ${CONDA_ENV_PREFIX}/bin/activate"
  # shellcheck source=/dev/null
  source "${CONDA_ENV_PREFIX}/bin/activate"
  log "Activated unpacked conda environment: ${CONDA_ENV_PREFIX}"
  if [[ -x "${CONDA_ENV_PREFIX}/bin/conda-unpack" ]]; then
    "${CONDA_ENV_PREFIX}/bin/conda-unpack"
  elif command -v conda-unpack >/dev/null 2>&1; then
    conda-unpack
  else
    die "conda-unpack not found after activating ${CONDA_ENV_PREFIX}"
  fi
  python -V
}

install_moxing_from_obs() {
  if enabled "${INSTALL_MOXING_FROM_OBS:-1}"; then
    install_moxing_wheel_from_obs "${REPO_ROOT}/${MOXING_WHEEL_NAME}" "[runtime]"
    log "MoXing file API is available after wheel installation."
  else
    log "INSTALL_MOXING_FROM_OBS=0, skip MoXing wheel installation."
  fi
}

register_local_packages() {
  [[ -f "${NUPLAN_DEVKIT_PACKAGE}" || -d "${NUPLAN_DEVKIT_PACKAGE}" ]] || die "nuPlan devkit package not found: ${NUPLAN_DEVKIT_PACKAGE}"
  [[ -d "${NAVSIM_DEVKIT_ROOT}" ]] || die "NAVSIM devkit directory not found: ${NAVSIM_DEVKIT_ROOT}"
  python -m pip install "${NUPLAN_DEVKIT_PACKAGE}" --no-deps
  python -m pip install -e "${NAVSIM_DEVKIT_ROOT}" --no-deps
  python -m pip install -e "${REPO_ROOT}" --no-deps
}

verify_runtime_environment() {
  python - "${REPO_ROOT}" <<'PY'
import sys
from pathlib import Path
repo_src = Path(sys.argv[1]) / "src"
if repo_src.is_dir():
    sys.path.insert(0, str(repo_src))
import fastwam, navsim, nuplan
import moxing as mox
if not hasattr(mox, "file"):
    raise RuntimeError("moxing is importable, but mox.file is unavailable")
print("runtime imports:")
print("  fastwam:", fastwam.__file__)
print("  navsim:", navsim.__file__)
print("  nuplan:", nuplan.__file__)
print("  moxing:", mox.__file__)
PY
}

sync_checkpoints() {
  mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}"
  local action_dit="${DIFFSYNTH_MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
  if enabled "${SYNC_CHECKPOINTS:-1}"; then
    if [[ -f "${action_dit}" ]] && ! enabled "${FORCE_SYNC_CHECKPOINTS:-0}"; then
      log "Checkpoint directory already contains ActionDiT; skip OBS checkpoint sync."
    else
      mox_copy_parallel "${OBS_CHECKPOINTS_ROOT%/}/" "${DIFFSYNTH_MODEL_BASE_PATH%/}/" 1
    fi
  fi
  [[ -f "${action_dit}" ]] || die "Missing ActionDiT checkpoint: ${action_dit}"
}

sync_or_build_text_embeds() {
  local local_root="${REPO_ROOT}/data/text_embeds_cache" navsim_cache="${local_root}/navsim_v1"
  mkdir -p "${local_root}"
  if enabled "${SYNC_TEXT_EMBEDS:-1}" && mox_path_exists "${OBS_TEXT_EMBEDS_ROOT}"; then
    mox_copy_parallel "${OBS_TEXT_EMBEDS_ROOT%/}/" "${local_root%/}/" 0
  fi
  if [[ ! -d "${navsim_cache}" ]] && enabled "${PRECOMPUTE_TEXT_EMBEDS:-1}"; then
    python scripts/precompute_text_embeds.py "task=${TASK}" "data.storage.mode=obs" "data.storage.obs_root=${OBS_DATA_ROOT}" "overwrite=false"
    enabled "${UPLOAD_TEXT_EMBEDS:-1}" && [[ -d "${navsim_cache}" ]] && mox_copy_parallel "${local_root%/}/" "${OBS_TEXT_EMBEDS_ROOT%/}/" 0
  fi
  [[ -d "${navsim_cache}" ]] || die "Missing text embedding cache: ${navsim_cache}"
}

obs_smoke_check() {
  mox_list_directory "${OBS_DATA_ROOT}"
  mox_list_directory "${OBS_DATA_ROOT%/}/navsim_logs/trainval"
}

run_training() {
  local train_args=("task=${TASK}" "data.storage.mode=obs" "data.storage.obs_root=${OBS_DATA_ROOT}")
  if enabled "${SMOKE}"; then
    train_args+=("data.train.max_scenes=2" "data.val.max_scenes=2" "eval_full_dataset=false" "max_steps=2" "num_epochs=1" "save_every=1" "eval_every=1" "wandb.enabled=false")
  fi
  train_args+=("$@")
  bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${train_args[@]}"
}

find_latest_checkpoint() {
  local weights_dir="${RUN_OUTPUT_DIR}/checkpoints/weights" checkpoint=""
  [[ -d "${weights_dir}" ]] || die "Checkpoint weights directory not found: ${weights_dir}"
  checkpoint="$(find "${weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)"
  [[ -n "${checkpoint}" ]] || die "No step_*.pt checkpoint found in ${weights_dir}"
  printf '%s\n' "${checkpoint}"
}

run_navtest_evaluation() {
  enabled "${RUN_NAVTEST_EVAL:-1}" || { log "RUN_NAVTEST_EVAL=0, skip navtest evaluation."; return 0; }
  local latest_checkpoint latest_step eval_output_dir stats_path
  latest_checkpoint="$(find_latest_checkpoint)"
  latest_step="$(basename "${latest_checkpoint}" .pt)"
  eval_output_dir="${RUN_OUTPUT_DIR}/eval_navtest_${latest_step}"
  stats_path="${RUN_OUTPUT_DIR}/dataset_stats.json"
  local eval_args=("task=${TASK}" "data.storage.mode=obs" "data.storage.obs_root=${OBS_DATA_ROOT}" "resume=${latest_checkpoint}" "output_dir=${eval_output_dir}")
  [[ -f "${stats_path}" ]] && eval_args+=("data.test.pretrained_norm_stats=${stats_path}")
  enabled "${SMOKE}" && eval_args+=("data.test.max_scenes=4" "+test_visualization.num_samples=4")
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/evaluate_navsim.py "${eval_args[@]}"
}

main() {
  print_runtime_info
  prepare_conda_env
  install_moxing_from_obs
  register_local_packages
  verify_runtime_environment
  sync_checkpoints
  sync_or_build_text_embeds
  obs_smoke_check
  run_training "$@"
  run_navtest_evaluation
  log "DI NavSIM run finished: ${RUN_OUTPUT_DIR}"
}
main "$@"

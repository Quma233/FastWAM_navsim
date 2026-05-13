# FastWAM NavSIM

This repository adapts FastWAM to NavSIM v1 SFT for autonomous driving. It keeps the original FastWAM model path intact and adds NavSIM data loading, full validation, visualization, and navtest evaluation utilities.

The current NavSIM task uses:

- `configs/model/fastwam.yaml`, not `fastwam_joint` or `fastwam_idm`.
- `CAM_F0` only.
- Current frame as the image condition.
- Current plus 8 future frames as the video tensor.
- 4 seconds / 8 steps of future ego trajectory as the action target.
- Full validation by default for NavSIM.

Large runtime artifacts are intentionally not tracked in git. Download or regenerate datasets, maps, metric caches, Wan2.2/T5/VAE weights, ActionDiT checkpoints, text embedding caches, and training runs on each training server.

## Layout

```text
FastWAM_navsim/
├── configs/
│   ├── data/navsim_v1.yaml
│   ├── model/fastwam.yaml
│   ├── task/navsim_v1_uncond_camf0_352x640_1e-4.yaml
│   └── train.yaml
├── requirements/fastwam_navsim_env.txt
├── scripts/
│   ├── evaluate_navsim.py
│   ├── precompute_text_embeds.py
│   ├── preprocess_action_dit_backbone.py
│   ├── run_di_navsim_obs.sh
│   ├── setup_navsim_env.sh
│   ├── train.py
│   └── train_zero1.sh
├── src/fastwam/
│   ├── datasets/navsim_io.py
│   ├── datasets/navsim_v1.py
│   ├── trainer.py
│   └── utils/navsim_visualization.py
├── third_party/
│   ├── navsim/
│   └── nuplan-devkit-v1.2.tar.gz
├── checkpoints/      # ignored
├── data/             # ignored except placeholder files
└── runs/             # ignored
```

## Environment Setup

Create and activate a Python 3.10 conda environment first:

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam

cd /path/to/FastWAM_navsim
bash scripts/setup_navsim_env.sh
```

`pip install -e .` alone is not enough for a fresh machine. The setup script installs:

- the vendored nuPlan devkit from `third_party/nuplan-devkit-v1.2.tar.gz`
- the vendored NAVSIM devkit from `third_party/navsim`
- the tested package set from `requirements/fastwam_navsim_env.txt`
- this repository in editable mode with `--no-deps`

The lock file was exported from the working `fastwam` conda environment. It avoids letting pip upgrade torch to a different CUDA build. The expected key versions are:

```text
python        3.10
torch         2.7.1+cu128
torchvision   0.22.1+cu128
numpy         1.26.4
hydra-core    1.3.2
accelerate    1.12.0
deepspeed     0.18.5
navsim        1.1.0
nuplan-devkit 1.2.0
```

To use a different CUDA wheel index:

```bash
CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_navsim_env.sh
```

To use external devkit copies instead of the vendored ones:

```bash
NAVSIM_DEVKIT_ROOT=/path/to/navsim \
NUPLAN_DEVKIT_PACKAGE=/path/to/nuplan-devkit-v1.2.tar.gz \
bash scripts/setup_navsim_env.sh
```

## Runtime Paths

Set these before local preprocessing, training, or evaluation:

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim_dataset/dataset
export NUPLAN_MAPS_ROOT=/path/to/navsim_dataset/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/navsim_dataset/exp
export NAVSIM_DEVKIT_ROOT=$(pwd)/third_party/navsim
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
```

For OBS-backed training, `OPENSCENE_DATA_ROOT` and `NAVSIM_EXP_ROOT` are not used for NavSIM logs, sensor blobs, or metric cache. Use `data.storage.mode=obs` and `data.storage.obs_root=...` instead. `NUPLAN_MAPS_ROOT` is still useful for local map access if any downstream NAVSIM code needs it.

## Model Files

This repo does not include model weights. Put Wan2.2/T5/VAE weights under `checkpoints/` using the same directory layout expected by the original FastWAM loader.

Generate the ActionDiT backbone checkpoint once per machine:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The NavSIM task expects:

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

## Data Interface

The NavSIM adapter is implemented in `src/fastwam/datasets/navsim_v1.py`.

Default data behavior:

- local mode uses NAVSIM official `SceneFilter`, `SensorConfig`, and `SceneLoader` style loading.
- OBS mode mirrors the required scene filtering in `NavsimV1FastWAMDataset` and reads pkl/image/metric-cache bytes through `mox.file`.
- Official NAVSIM/Drive-JEPA split files, not a fixed sample ratio.
- `CAM_F0` only.
- Video shape `[3, 9, 352, 640]`.
- Action shape `[8, 3]`.
- Proprio shape `[1, 8]` from `driving_command(4) + velocity(2) + acceleration(2)`.
- Image preprocessing: `image[28:-28]`, resize to `[352, 640]`, then FastWAM `Normalize(0.5, 0.5)`.

Current train/val defaults in `configs/data/navsim_v1.yaml`:

```yaml
storage:
  mode: local  # local | obs
  obs_root: obs://yw-2030-gy/external/personal/f50000365/navsim_dataset
  local_dataset_root: ${oc.env:OPENSCENE_DATA_ROOT,/cache/navsim_dataset/dataset}
  local_exp_root: ${oc.env:NAVSIM_EXP_ROOT,/cache/navsim_dataset/exp}
  obs_image_cache_size: 128
  obs_filter_missing_images: false

train:
  navsim_log_path: ${navsim_storage_path:${data.storage.mode},${data.storage.local_dataset_root},${data.storage.obs_root},navsim_logs/trainval}
  sensor_blobs_path: ${navsim_storage_path:${data.storage.mode},${data.storage.local_dataset_root},${data.storage.obs_root},sensor_blobs/trainval}
  metric_cache_path: ${navsim_storage_path:${data.storage.mode},${data.storage.local_exp_root},${data.storage.obs_root}/exp,metric_cache_v1.1.0_navtrain}
  storage_mode: ${data.storage.mode}
  split: train
  split_mode: official
  navsim_split_name: navtrain  # all_scenes
  log_split_file: default_train_val_test_log_split.yaml

val:
  navsim_log_path: ${data.train.navsim_log_path}
  sensor_blobs_path: ${data.train.sensor_blobs_path}
  metric_cache_path: ${data.train.metric_cache_path}
  storage_mode: ${data.train.storage_mode}
  split: val
  navsim_split_name: ${data.train.navsim_split_name}

test:
  navsim_log_path: ${navsim_storage_path:${data.storage.mode},${data.storage.local_dataset_root},${data.storage.obs_root},navsim_logs/test}
  sensor_blobs_path: ${navsim_storage_path:${data.storage.mode},${data.storage.local_dataset_root},${data.storage.obs_root},sensor_blobs/test}
  metric_cache_path: ${navsim_storage_path:${data.storage.mode},${data.storage.local_exp_root},${data.storage.obs_root}/exp,metric_cache_v1.1.0_navtest}
  storage_mode: ${data.storage.mode}
  split: test
  navsim_split_name: navtest
```

`navsim_split_name: navtrain` matches the current training setup. `all_scenes` is left as a commented alternative for experiments where you intentionally want that NAVSIM scene filter.

For a mini-set debug run without editing YAML:

```bash
bash scripts/train_zero1.sh 2 task=navsim_v1_uncond_camf0_352x640_1e-4 \
  data.train.navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/mini \
  data.train.sensor_blobs_path=$OPENSCENE_DATA_ROOT/sensor_blobs/mini
```

## OBS Online Data Mode

NavSIM data can be read in two modes:

- `local`: the default, using regular filesystem paths under `OPENSCENE_DATA_ROOT` and `NAVSIM_EXP_ROOT`.
- `obs`: online reads through Huawei MoXing. Logs, images, and metric cache are read from OBS without copying the full NavSIM dataset to `/cache`.

Switch to OBS with a task override or command-line override:

```bash
bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4 \
  data.storage.mode=obs \
  data.storage.obs_root=obs://yw-2030-gy/external/personal/f50000365/navsim_dataset
```

The OBS root must match this structure:

```text
navsim_dataset/
  navsim_logs/...
  sensor_blobs/...
  exp/metric_cache_v1.1.0_navtrain/...
  exp/metric_cache_v1.1.0_navtest/...
```

OBS mode requires the official ModelArts MoXing package with `mox.file`, plus these S3-compatible OBS variables:

```bash
export S3_ENDPOINT=https://obs.cn-southwest-2.myhuaweicloud.com
export S3_USE_HTTPS=0
export ACCESS_KEY_ID=...
export SECRET_ACCESS_KEY=...
```

Do not install the public PyPI package named `moxing`; it does not provide the `mox.file` API used here. Use the official ModelArts `moxing_framework-*.whl` when the runtime image does not already include MoXing.

## Text Embedding Cache

NavSIM uses one fixed prompt:

```text
A front camera video from an autonomous vehicle. Predict the ego vehicle's 4-second future trajectory.
```

Precompute it before training:

```bash
python scripts/precompute_text_embeds.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  overwrite=false
```

Default cache path:

```text
data/text_embeds_cache/navsim_v1
```

This cache is ignored by git.

## Training Config

The NavSIM task config is `configs/task/navsim_v1_uncond_camf0_352x640_1e-4.yaml`.

Current full-training defaults:

```yaml
batch_size: 16
num_workers: 16

model:
  mot_checkpoint_mixed_attn: false
  loss:
    lambda_video: 0.1
    lambda_action: 1.0

num_epochs: 50
save_every: 1774
eval_every: 1774
eval_full_dataset: true

eval_visualization:
  enabled: true
  num_samples: 32
  world_model: true
  trajectory: true

optimizer_impl: deepspeed_fused_adam
weight_decay: 1e-2

wandb:
  enabled: true
  project: fastwam_navsim
  group: navsim_sft_v1
  mode: offline
```

`save_every` and `eval_every` are step intervals, not epoch counts. Set them to the number of optimizer steps in one epoch if you want one save/eval per epoch.

Global defaults such as learning rate, scheduler, precision, and gradient clipping live in `configs/train.yaml`. The NavSIM task overrides the fields shown above, including the video/action loss weights.

## Training

Example for 8 GPUs:

```bash
bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

Example for 2 GPUs:

```bash
bash scripts/train_zero1.sh 2 task=navsim_v1_uncond_camf0_352x640_1e-4
```

Useful overrides:

```bash
bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4 \
  batch_size=8 \
  num_workers=12 \
  eval_every=2000 \
  save_every=2000
```

The launcher writes terminal output to:

```text
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/train.log
```

To choose a run id:

```bash
RUN_ID=my_full_train_001 bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

To disable terminal log capture for a temporary run:

```bash
TRAIN_LOG_TO_FILE=0 bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

Run outputs include:

```text
runs/<task>/<RUN_ID>/config.yaml
runs/<task>/<RUN_ID>/dataset_stats.json
runs/<task>/<RUN_ID>/checkpoints/
runs/<task>/<RUN_ID>/eval/navsim_step_*.csv
runs/<task>/<RUN_ID>/eval/vis/
```

## DI Platform Run Script

`scripts/run_di_navsim_obs.sh` is the single-node 8-GPU launcher for the DI platform. It first syncs the repo from OBS to a local workspace, then uses the synced repo-local conda-pack environment that already includes MoXing, OBS-hosted checkpoints, OBS online NavSIM data, and runs navtest evaluation after training.

Expected repo-local environment directory:

```text
conda_envs/fastwam_moxing_di/
  bin/python
  bin/activate
  bin/conda-unpack
```

Default local repo sync target:

```text
${WORKSPACE:-/home/ma-user/code}/FastWAM_navsim_di
```

At startup, if the script is not already running from this target and `SYNC_REPO_FROM_OBS` is not `0`, it uses the launch environment's `moxing.file.copy_parallel` to copy `${OBS_REPO_ROOT}/` into that local directory, then re-executes the synced script. This mirrors the DI sample flow of copying code from OBS into `WORKSPACE` before running training. Because this bootstrap happens before the repo-local conda environment exists locally, the initial launcher environment must already provide `moxing.file`.

Place the complete unpacked environment directory under `conda_envs/fastwam_moxing_di`. This environment is expected to already contain FastWAM runtime dependencies, NAVSIM/nuPlan dependencies, and a working MoXing package with `mox.file`. The DI launcher activates it, runs `conda-unpack`, downloads `moxing_framework-2.5.0rc6-py2.py3-none-any.whl` from OBS, installs that wheel with `pip install ... --upgrade-strategy only-if-needed`, then re-registers the repo-local Python packages with `pip install --no-deps` so editable/package paths point at the current checkout.

Expected OBS layout:

```text
obs://yw-2030-gy/external/personal/f50000365/FastWAM_navsim_di/
  checkpoints/
  data/text_embeds_cache/
  runs/

obs://yw-2030-gy/external/personal/f50000365/navsim_dataset/
  navsim_logs/...
  sensor_blobs/...
  exp/metric_cache_v1.1.0_navtrain/...
  exp/metric_cache_v1.1.0_navtest/...
```

Default run:

```bash
bash scripts/run_di_navsim_obs.sh
```

The script defaults to 8 GPUs. A numeric first argument is also accepted:

```bash
bash scripts/run_di_navsim_obs.sh 8
```

Smoke run:

```bash
SMOKE=1 bash scripts/run_di_navsim_obs.sh
```

Useful overrides:

```bash
RUN_ID=my_di_run_001 \
OBS_REPO_ROOT=obs://yw-2030-gy/external/personal/f50000365/FastWAM_navsim_di \
OBS_DATA_ROOT=obs://yw-2030-gy/external/personal/f50000365/navsim_dataset \
bash scripts/run_di_navsim_obs.sh \
  batch_size=8 \
  num_workers=12
```

The script:

- syncs `${OBS_REPO_ROOT}/` into `${WORKSPACE:-/home/ma-user/code}/FastWAM_navsim_di` before training, unless already running there or `SYNC_REPO_FROM_OBS=0`
- requires the complete environment at `conda_envs/fastwam_moxing_di` inside the synced repo
- activates `conda_envs/fastwam_moxing_di`, then runs `conda-unpack`
- downloads `obs://yw-ads-training-gy1/data/external/personal/z00009214/moxing_framework-2.5.0rc6-py2.py3-none-any.whl` with `moxing.file.copy`, then installs it with `pip install ... --upgrade-strategy only-if-needed`; set `INSTALL_MOXING_FROM_OBS=0` to skip this step
- does not run `scripts/setup_navsim_env.sh` or reinstall `requirements/fastwam_navsim_env.txt`
- re-registers `third_party/nuplan-devkit-v1.2.tar.gz`, `third_party/navsim`, and the current repo with `pip install ... --no-deps`
- verifies that current repo `fastwam`, vendored `navsim`/`nuplan`, and environment `moxing.file` are importable
- uses local `checkpoints/` from the synced repo, or syncs `${OBS_REPO_ROOT}/checkpoints/` if the key ActionDiT checkpoint is missing; set `FORCE_SYNC_CHECKPOINTS=1` to force resync
- syncs or precomputes `data/text_embeds_cache/navsim_v1`
- launches `scripts/train_zero1.sh 8 ... data.storage.mode=obs`
- finds the newest `runs/<task>/<RUN_ID>/checkpoints/weights/step_*.pt`
- runs `torchrun --standalone --nproc_per_node=8 scripts/evaluate_navsim.py ...`
- uploads the whole run directory to `${OBS_REPO_ROOT}/runs/<task>/<RUN_ID>/`

The script contains the OBS credentials from the DI sample script and does not print a full `env` dump. Override `ACCESS_KEY_ID` or `SECRET_ACCESS_KEY` externally if you need different credentials.

## Validation

For NavSIM, `eval_full_dataset: true` makes each validation event iterate through the full validation split. It reports:

- `val_loss`
- `traj_l1`
- `ade`
- `fde`
- `ade_2s`
- `fde_2s`
- `heading_mae`

If a valid metric cache exists, training validation also reports NAVSIM PDM metrics with `pdm_` prefixes, for example `pdm_score`.

PDM validation is skipped without blocking training when:

- `metric_cache_path` does not exist
- `metric_cache_path/metadata` does not exist
- a validation token is missing from the metric cache

Per-token validation rows are saved to:

```text
runs/<task>/<RUN_ID>/eval/navsim_step_*.csv
```

## Validation Visualizations

The current task saves visualizations for 32 validation samples per eval event:

```yaml
eval_visualization:
  enabled: true
  num_samples: 32
  world_model: true
  trajectory: true
```

The selected visualization samples are the first `num_samples` dataset indices:

```text
idx = 0, 1, ..., num_samples - 1
```

Outputs:

```text
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/index.csv
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/world_model/*.png
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/trajectory/*.png
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/bev/*.png
```

World-model images show predicted future frames on the left and GT future frames on the right.

Trajectory images show the current `CAM_F0` frame with:

- green: GT future trajectory
- red: predicted future trajectory

BEV images show a simple ego-frame top-down view without map or annotations:

- green: GT future trajectory
- red: predicted future trajectory
- black box: ego vehicle

`psnr_rg` and `ssim_rg` are computed only on these selected visualization samples, not over the whole validation set. They are logged as image-quality diagnostics for the world-model output.

## NavSIM Test Evaluation

Use `scripts/evaluate_navsim.py` to load a trained checkpoint and evaluate on the NavSIM test split:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/navsim_v1_uncond_camf0_352x640_1e-4/eval_navtest_step_XXXXXX
```

Multi-GPU evaluation is supported with `torchrun`; each rank evaluates a shard of the test set and rank 0 writes the merged CSV/submission/visualizations:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/navsim_v1_uncond_camf0_352x640_1e-4/eval_navtest_step_XXXXXX
```

Do not pass `eval_device` for multi-GPU evaluation; the script assigns `cuda:<LOCAL_RANK>` automatically.

The script:

- builds `cfg.data.test`
- loads `dataset_stats.json` automatically from the checkpoint run directory when possible
- runs `model.infer_action`
- writes per-token metrics plus an average row
- writes a NAVSIM-style submission pickle
- saves test visualizations when `eval_visualization.enabled=true`, or when `test_visualization` is provided
- prints the final average metrics to the terminal

Outputs:

```text
navsim_test_metrics.csv
submission.pkl
eval_config.yaml
vis/index.csv
vis/world_model/*.png
vis/trajectory/*.png
vis/bev/*.png
```

`navsim_test_metrics.csv` uses a compact readable format:

```text
idx,token,log_name,metrics
0,<token>,<log>,"ade: ...; fde: ...; score: ..."
1,<token>,<log>,"ade: ...; fde: ...; score: ..."
-1,average,,"ade: ...; fde: ...; score: ..."
```

The terminal only prints the final average metrics, not one line per sample.

By default, the current task config enables 32 visualization samples through `eval_visualization`. These are the first 32 test samples. Test visualization uses the same fields unless you add a test-only override:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_step_XXXXXX \
  +test_visualization.enabled=true \
  +test_visualization.num_samples=32 \
  +test_visualization.world_model=true \
  +test_visualization.trajectory=true \
  +test_visualization.bev=true
```

To quickly visualize only the first 4 test samples:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_vis4 \
  +test_visualization.num_samples=4
```

To disable visualization for a metrics-only test run:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_step_XXXXXX \
  eval_visualization.enabled=false
```

To disable only BEV visualization:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_step_XXXXXX \
  +test_visualization.bev=false
```

## Bad Sample Filtering

After running `scripts/evaluate_navsim.py` with visualization enabled, you can collect low-PDM samples into per-sample folders without rerunning inference:

```bash
python scripts/filter_bad_navsim_samples.py \
  --metrics_csv runs/.../eval_navtest_step_060316/navsim_test_metrics.csv \
  --vis_index_csv runs/.../eval_navtest_step_060316/vis/index.csv \
  --metric score \
  --threshold 0.3
```

The script reads the test metrics CSV and the visualization index:

```text
runs/.../eval_navtest_step_060316/navsim_test_metrics.csv
runs/.../eval_navtest_step_060316/vis/index.csv
```

`score` is the NAVSIM PDM score in test output. The script selects samples with `score < threshold` by default. It refreshes the output directory on every non-dry run, so repeated runs do not mix old and new bad-sample results. Each selected sample is copied to:

```text
bad_samples/<token>_<metric>_<value>/
```

Each folder contains the existing visualizations:

```text
bev.png
camera_trajectory.png
future_rollout.png
metrics.txt
```

Use `--comparison le`, `--max_samples`, `--output_dir`, or `--dry_run` when needed. If you accidentally pass `--metric pdm_score` on test output, the script aliases it to `score` when that field exists.

Test evaluation requires `data.test.metric_cache_path`. By default it is:

```text
${NAVSIM_EXP_ROOT}/metric_cache_v1.1.0_navtest
```

If your checkpoint is outside its original run directory, pass the normalization stats explicitly:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/step_XXXXXX.pt \
  data.test.pretrained_norm_stats=/path/to/dataset_stats.json \
  output_dir=./runs/eval_navtest_step_XXXXXX
```

## Metric Cache

Metric cache is a NAVSIM artifact needed for PDM scoring. It is not included in this repo.

Example command for navtrain/trainval cache:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=trainval \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_v1.1.0_navtrain \
  worker=single_machine_thread_pool \
  worker.max_workers=32
```

Example command for navtest cache:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_v1.1.0_navtest \
  worker=single_machine_thread_pool \
  worker.max_workers=32
```

`worker=single_machine_thread_pool` and `worker.max_workers=32` are standard NAVSIM Hydra overrides used for local parallel cache generation. Adjust `worker.max_workers` for your CPU and memory budget.

## Git Hygiene

Do not commit large runtime files:

```text
checkpoints/
conda_envs/
runs/
data/text_embeds_cache/
*.pt
*.safetensors
*.pkl
*.pickle
*.bin
```

Exception: `third_party/nuplan-devkit-v1.2.tar.gz` is intentionally tracked so restricted training servers do not need to clone nuPlan.

Before pushing:

```bash
git status --short
git diff --cached --name-only
```

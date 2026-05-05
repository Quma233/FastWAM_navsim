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
│   ├── setup_navsim_env.sh
│   ├── train.py
│   └── train_zero1.sh
├── src/fastwam/
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

Set these before preprocessing, training, or evaluation:

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim_dataset/dataset
export NUPLAN_MAPS_ROOT=/path/to/navsim_dataset/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/navsim_dataset/exp
export NAVSIM_DEVKIT_ROOT=$(pwd)/third_party/navsim
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
```

`configs/data/navsim_v1.yaml` contains placeholder fallbacks under `/path/to/navsim_dataset/...`. Prefer environment variables or Hydra overrides over committing machine-specific absolute paths.

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

- NAVSIM official `SceneFilter`, `SensorConfig`, and `SceneLoader` style loading.
- Official NAVSIM/Drive-JEPA split files, not a fixed sample ratio.
- `CAM_F0` only.
- Video shape `[3, 9, 352, 640]`.
- Action shape `[8, 3]`.
- Proprio shape `[1, 8]` from `driving_command(4) + velocity(2) + acceleration(2)`.
- Image preprocessing: `image[28:-28]`, resize to `[352, 640]`, then FastWAM `Normalize(0.5, 0.5)`.

Current train/val defaults in `configs/data/navsim_v1.yaml`:

```yaml
train:
  navsim_log_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/navsim_logs/trainval
  sensor_blobs_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/sensor_blobs/trainval
  metric_cache_path: ${oc.env:NAVSIM_EXP_ROOT,/path/to/navsim_dataset/exp}/metric_cache_v1.1.0_navtrain
  split: train
  split_mode: official
  navsim_split_name: navtrain  # all_scenes
  log_split_file: default_train_val_test_log_split.yaml

val:
  navsim_log_path: ${data.train.navsim_log_path}
  sensor_blobs_path: ${data.train.sensor_blobs_path}
  metric_cache_path: ${data.train.metric_cache_path}
  split: val
  navsim_split_name: ${data.train.navsim_split_name}

test:
  navsim_log_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/navsim_logs/test
  sensor_blobs_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/sensor_blobs/test
  metric_cache_path: ${oc.env:NAVSIM_EXP_ROOT,/path/to/navsim_dataset/exp}/metric_cache_v1.1.0_navtest
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

Global defaults such as learning rate, scheduler, precision, gradient clipping, and loss weights live in `configs/train.yaml`. The NavSIM task only overrides the fields shown above.

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

Outputs:

```text
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/index.csv
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/world_model/*.png
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/trajectory/*.png
```

World-model images show predicted future frames on the left and GT future frames on the right.

Trajectory images show the current `CAM_F0` frame with:

- green: GT future trajectory
- red: predicted future trajectory

`psnr_rg` and `ssim_rg` are computed only on these selected visualization samples, not over the whole validation set. They are logged as image-quality diagnostics for the world-model output.

## NavSIM Test Evaluation

Use `scripts/evaluate_navsim.py` to load a trained checkpoint and evaluate on the NavSIM test split:

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/navsim_v1_uncond_camf0_352x640_1e-4/eval_navtest_step_XXXXXX
```

The script:

- builds `cfg.data.test`
- loads `dataset_stats.json` automatically from the checkpoint run directory when possible
- runs `model.infer_action`
- writes per-token metrics and an average row
- writes a NAVSIM-style submission pickle

Outputs:

```text
navsim_test_metrics.csv
submission.pkl
eval_config.yaml
```

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

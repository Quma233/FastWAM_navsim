# FastWAM NavSIM

This repository is a NavSIM v1 SFT adaptation of FastWAM for autonomous driving trajectory prediction.

It keeps the original FastWAM model path and training logic as much as possible:

- Model config uses `configs/model/fastwam.yaml`.
- NavSIM uses the default FastWAM model, not `joint` or `idm`.
- FastWAM model code is not changed for the NavSIM data adapter.
- NavSIM validation runs the full validation dataset by default.

The NAVSIM and nuPlan devkits are vendored under `third_party/` so the training server does not need to clone those repositories. Large runtime artifacts such as datasets, maps, metric caches, Wan2.2 checkpoints, ActionDiT checkpoints, text embedding caches, and training outputs are intentionally not stored in git.

## Repository Layout

```text
FastWAM_navsim/
├── configs/
│   ├── data/navsim_v1.yaml
│   ├── model/fastwam.yaml
│   ├── task/navsim_v1_uncond_camf0_352x640_1e-4.yaml
│   └── train.yaml
├── scripts/
│   ├── preprocess_action_dit_backbone.py
│   ├── precompute_text_embeds.py
│   ├── train.py
│   └── train_zero1.sh
├── src/fastwam/
│   ├── datasets/navsim_v1.py
│   └── trainer.py
├── third_party/
│   ├── navsim/                         # vendored NAVSIM devkit
│   └── nuplan-devkit-v1.2.tar.gz       # vendored nuPlan devkit package
├── checkpoints/      # ignored, put downloaded/generated model weights here
├── data/             # ignored, text embedding cache can be generated here
└── runs/             # ignored, training logs/checkpoints/eval CSVs are saved here
```

## NavSIM Data Interface

The NavSIM dataset adapter is implemented in:

```text
src/fastwam/datasets/navsim_v1.py
```

Default behavior:

- NAVSIM official `SceneFilter`, `SensorConfig`, and `SceneLoader` style data loading.
- Default train/val/test split follows the NAVSIM/Drive-JEPA official log and token split configs, not a fixed sample ratio.
- Camera: `CAM_F0` only.
- Input image: current frame only.
- Video tensor: current frame plus future 8 frames, total 9 frames.
- Action/trajectory target: future 4 seconds, 8 frames, shape `[8, 3]`.
- Proprio/current ego state: `driving_command(4) + velocity(2) + acceleration(2)`, shape `[1, 8]`.
- Image preprocessing: crop `image[28:-28]`, resize to `[352, 640]`, then FastWAM normalization `Normalize(0.5, 0.5)`.
- Text prompt is fixed for NavSIM and cached before training.

## Environment

This repo includes the NAVSIM and nuPlan devkits needed by the code:

```text
third_party/navsim
third_party/nuplan-devkit-v1.2.tar.gz
```

`pip install -e .` alone only installs FastWAM and its Python package metadata. Use the setup script below so the vendored devkits are installed first, then the exact package versions from `requirements/fastwam_navsim_env.txt` are installed, and finally this repo is installed in editable mode without dependency resolution.

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam

cd /path/to/FastWAM_navsim
bash scripts/setup_navsim_env.sh
```

The lock file `requirements/fastwam_navsim_env.txt` was exported from the working `fastwam` conda environment. It intentionally excludes local editable installs for FastWAM, NAVSIM, and nuPlan because those are installed from this repo's source tree. The vendored NAVSIM `setup.py` keeps upstream `requirements.txt` for reference but does not publish those old pins as package metadata; the locked environment controls dependency versions.

The tested key versions are:

```text
python              3.10
torch               2.7.1+cu128
torchvision         0.22.1+cu128
numpy               1.26.4
hydra-core          1.3.2
accelerate          1.12.0
deepspeed           0.18.5
navsim              1.1.0
nuplan-devkit       1.2.0
opencv-python       4.9.0.80
scikit-learn        1.2.2
positional-encodings 6.0.1
pytorch-lightning   2.2.1
tensorboard         2.16.2
protobuf            4.25.3
```

If you need a different CUDA wheel index, override it before running the script:

```bash
CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_navsim_env.sh
```

If you want to use an external NAVSIM or nuPlan devkit instead of the vendored copies, override these paths:

```bash
NAVSIM_DEVKIT_ROOT=/path/to/navsim \
NUPLAN_DEVKIT_PACKAGE=/path/to/nuplan-devkit-v1.2.tar.gz \
bash scripts/setup_navsim_env.sh
```

After installation, set the runtime paths before preprocessing or training. These paths still point to your local dataset, maps, and experiment/cache directory; they are not included in this repo.

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim_dataset/dataset
export NUPLAN_MAPS_ROOT=/path/to/navsim_dataset/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/navsim_dataset/exp
export NAVSIM_DEVKIT_ROOT=$(pwd)/third_party/navsim
```

`configs/data/navsim_v1.yaml` contains placeholder fallback paths under `/path/to/navsim_dataset/...`. In normal use, set the environment variables above instead of editing absolute machine-specific paths into the repo.

## Model Preparation

This repository does not upload large pretrained weights. Prepare them again on the training server.

Set the model cache root:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Generate the ActionDiT backbone checkpoint:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The training config expects this file at:

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

Wan2.2 / T5 / VAE weights are also expected under `checkpoints/` through the same loading path used by FastWAM.

## Optional: Generate NAVSIM Metric Cache for PDM

PDM metrics require NAVSIM metric cache. Without it, training still runs and validation still reports trajectory metrics such as `ADE`, `FDE`, `ADE_2s`, `FDE_2s`, `traj_l1`, and `heading_mae`.

For full `trainval` data:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=trainval \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  worker=single_machine_thread_pool \
  worker.max_workers=32
```

For the official mini split, use:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=mini \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  worker=single_machine_thread_pool \
  worker.max_workers=32
```

For the NAVSIM test split, use `train_test_split=navtest` and point the evaluation/test data to `navsim_logs/test` and `sensor_blobs/test`.

PDM metrics are computed only when:

- `eval_full_dataset: true`
- `metric_cache_path` exists
- `metric_cache_path/metadata` exists
- the evaluated token is present in the metric cache

## Precompute Text Embeddings

NavSIM uses a fixed prompt:

```text
A front camera video from an autonomous vehicle. Predict the ego vehicle's 4-second future trajectory.
```

Generate the text embedding cache before training:

```bash
python scripts/precompute_text_embeds.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  overwrite=false
```

The default cache directory is:

```text
data/text_embeds_cache/navsim_v1
```

This directory is ignored by git and should be regenerated on each new machine.

## Train

Run distributed ZeRO-1 training with:

```bash
bash scripts/train_zero1.sh <num_gpus> task=navsim_v1_uncond_camf0_352x640_1e-4
```

Example for 8 GPUs:

```bash
bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

Example for 2 GPUs:

```bash
bash scripts/train_zero1.sh 2 task=navsim_v1_uncond_camf0_352x640_1e-4
```

The script writes terminal output to both the console and:

```text
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/train.log
```

The run directory also contains:

```text
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/config.yaml
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/dataset_stats.json
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/checkpoints/
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/eval/navsim_step_*.csv
```

To choose a stable run id manually:

```bash
RUN_ID=my_full_train_001 bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

To disable log-to-file for a temporary run:

```bash
TRAIN_LOG_TO_FILE=0 bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

## Main Configuration Files

### Data paths and splits

Edit or override:

```text
configs/data/navsim_v1.yaml
```

Important fields:

```yaml
train:
  navsim_log_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/navsim_logs/trainval
  sensor_blobs_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/sensor_blobs/trainval
  metric_cache_path: ${oc.env:NAVSIM_EXP_ROOT,/path/to/navsim_dataset/exp}/metric_cache
  split: train
  split_mode: official
  navsim_split_name: navtrain
  split_config_root: ./third_party/navsim/navsim/planning/script/config
  log_split_file: default_train_val_test_log_split.yaml

val:
  navsim_log_path: ${data.train.navsim_log_path}
  sensor_blobs_path: ${data.train.sensor_blobs_path}
  split: val
  split_mode: ${data.train.split_mode}
  navsim_split_name: ${data.train.navsim_split_name}

test:
  navsim_log_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/navsim_logs/test
  sensor_blobs_path: ${oc.env:OPENSCENE_DATA_ROOT,/path/to/navsim_dataset/dataset}/sensor_blobs/test
  split: test
  split_mode: ${data.train.split_mode}
  navsim_split_name: navtest
```

By default, `split_mode: official` matches Drive-JEPA/NAVSIM v1 behavior:

- `train` and `val` both read `navsim_logs/trainval`, use the `navtrain` scene filter, then intersect its log/token set with official `train_logs` or `val_logs` from `default_train_val_test_log_split.yaml`.
- `test` reads `navsim_logs/test`, uses the `navtest` scene filter, and intersects with official `test_logs`.
- The old deterministic sample-ratio split is still available only as an explicit debug fallback with `split_mode=proportion split_proportion=0.95`.

To train on the current mini dataset without editing the YAML, point train/val to the mini folders. The official split logic will use the mini logs that overlap the official train/val log lists.

```bash
export OPENSCENE_DATA_ROOT=/data1/jcfu/navsim_dataset_v1/dataset
export NAVSIM_EXP_ROOT=/data1/jcfu/navsim_dataset_v1/exp

bash scripts/train_zero1.sh 2 task=navsim_v1_uncond_camf0_352x640_1e-4 \
  data.train.navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/mini \
  data.train.sensor_blobs_path=$OPENSCENE_DATA_ROOT/sensor_blobs/mini
```

For full training, keep the default `trainval` and `test` paths and only set `OPENSCENE_DATA_ROOT` to the full NAVSIM dataset root.

### Training settings

Edit:

```text
configs/task/navsim_v1_uncond_camf0_352x640_1e-4.yaml
```

Current full-training defaults:

```yaml
batch_size: 2
num_workers: 8
save_every: 1000
eval_every: 1000
eval_full_dataset: true
optimizer_impl: deepspeed_fused_adam
```

Common overrides:

```bash
bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4 \
  batch_size=4 \
  num_workers=12 \
  eval_every=2000 \
  save_every=2000
```

Other global training defaults live in:

```text
configs/train.yaml
```

Examples:

```yaml
learning_rate: 1.0e-4
weight_decay: 0.0
num_epochs: 1
max_steps: null
mixed_precision: bf16
gradient_accumulation_steps: 1
max_grad_norm: 1.0
eval_num_inference_steps: 10
```

### Model settings

NavSIM uses:

```text
configs/model/fastwam.yaml
```

This keeps the default FastWAM model path:

```yaml
mot_checkpoint_mixed_attn: true
action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

Do not switch to `fastwam_joint` or `fastwam_idm` unless intentionally changing the task design.

### Optimizer setting

NavSIM task uses:

```yaml
optimizer_impl: deepspeed_fused_adam
```

This keeps AdamW-style optimization while using DeepSpeed FusedAdam. It was chosen because PyTorch AdamW produced non-finite parameters in bf16 + ZeRO-1 testing for this setup. Learning rate, weight decay, betas, scheduler, losses, and model code remain unchanged.

## Validation Outputs

With `eval_full_dataset: true`, every validation event iterates over the full validation dataset and writes a per-token CSV:

```text
runs/<task>/<RUN_ID>/eval/navsim_step_*.csv
```

Always reported:

- `val_loss`
- `traj_l1`
- `ADE`
- `FDE`
- `ADE_2s`
- `FDE_2s`
- `heading_mae`

Reported only when NAVSIM metric cache is available:

- PDM `score`
- PDM submetrics returned by the NAVSIM scorer

## Files Intentionally Not Uploaded

These are ignored by git and should be regenerated or downloaded on each server:

```text
checkpoints/
runs/
data/*
*.pt
*.safetensors
*.pkl
*.pickle
*.bin
*.tar
*.tar.gz
```

Exception: `third_party/nuplan-devkit-v1.2.tar.gz` is intentionally tracked so restricted training machines do not need to clone the nuPlan devkit.

Before pushing to GitHub, verify staged files:

```bash
git diff --cached --name-only
```

Do not commit model weights, generated caches, dataset files, or training outputs.

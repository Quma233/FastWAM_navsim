# FastWAM NavSIM 使用说明

本仓库是 FastWAM 到 NavSIM v1 SFT 的自动驾驶迁移版本。模型仍使用原版 `configs/model/fastwam.yaml`，没有切换到 `joint/idm`，主要新增的是 NavSIM 数据接口、完整 val、可视化、以及 navtest 评估脚本。

## 环境安装

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam

cd /path/to/FastWAM_navsim
bash scripts/setup_navsim_env.sh
```

不要只运行 `pip install -e .`。`setup_navsim_env.sh` 会先安装仓库内的 `third_party/navsim` 和 `third_party/nuplan-devkit-v1.2.tar.gz`，再按 `requirements/fastwam_navsim_env.txt` 安装当前验证过的包版本，最后以 editable 方式安装本仓库。

## 运行前路径

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim_dataset/dataset
export NUPLAN_MAPS_ROOT=/path/to/navsim_dataset/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/path/to/navsim_dataset/exp
export NAVSIM_DEVKIT_ROOT=$(pwd)/third_party/navsim
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
```

大文件不在 git 里，包括 NavSIM 数据、maps、metric cache、Wan2.2/T5/VAE 权重、ActionDiT checkpoint、text embedding cache、训练 runs/checkpoints。

## 模型准备

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## 数据配置

主要文件是 `configs/data/navsim_v1.yaml`。

当前默认设置：

- train/val 读 `navsim_logs/trainval` 和 `sensor_blobs/trainval`
- test 读 `navsim_logs/test` 和 `sensor_blobs/test`
- train metric cache: `${NAVSIM_EXP_ROOT}/metric_cache_v1.1.0_navtrain`
- test metric cache: `${NAVSIM_EXP_ROOT}/metric_cache_v1.1.0_navtest`
- `split_mode: official`
- `navsim_split_name: navtrain  # all_scenes`
- 相机只用 `CAM_F0`
- 输入只用当前帧
- video 是当前帧加未来 8 帧，shape `[3, 9, 352, 640]`
- action 是未来 4s 的 8 帧轨迹，shape `[8, 3]`
- proprio 是 `driving_command(4)+velocity(2)+acceleration(2)`，shape `[1, 8]`

mini set 调试时可以不改 YAML，直接命令行 override：

```bash
bash scripts/train_zero1.sh 2 task=navsim_v1_uncond_camf0_352x640_1e-4 \
  data.train.navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/mini \
  data.train.sensor_blobs_path=$OPENSCENE_DATA_ROOT/sensor_blobs/mini
```

## 文本 embedding

NavSIM 使用固定 prompt：

```text
A front camera video from an autonomous vehicle. Predict the ego vehicle's 4-second future trajectory.
```

训练前运行：

```bash
python scripts/precompute_text_embeds.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  overwrite=false
```

## 当前训练参数

主要文件是 `configs/task/navsim_v1_uncond_camf0_352x640_1e-4.yaml`。

当前正式训练默认值：

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

`save_every` 和 `eval_every` 是 step 间隔，不是 epoch 数。如果想每个 epoch 保存和评估一次，把它们设成一个 epoch 对应的 optimizer step 数。

learning rate、scheduler、precision、gradient clipping 等全局默认值仍在 `configs/train.yaml`。NavSIM task 会覆盖上面列出的字段，包括 video/action loss weight。

## 训练命令

8 卡：

```bash
bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

2 卡：

```bash
bash scripts/train_zero1.sh 2 task=navsim_v1_uncond_camf0_352x640_1e-4
```

日志会同时输出到终端和：

```text
runs/navsim_v1_uncond_camf0_352x640_1e-4/<RUN_ID>/train.log
```

手动指定 run id：

```bash
RUN_ID=my_full_train_001 bash scripts/train_zero1.sh 8 task=navsim_v1_uncond_camf0_352x640_1e-4
```

## Val 输出

NavSIM 默认 `eval_full_dataset: true`，每次 val 会遍历完整 val dataset，并输出：

- `val_loss`
- `traj_l1`
- `ade`
- `fde`
- `ade_2s`
- `fde_2s`
- `heading_mae`

如果 metric cache 可用，还会输出 `pdm_` 前缀的 PDM 指标，例如 `pdm_score`。如果 cache 缺失或 token 不在 cache 里，PDM 会跳过，不会阻塞训练。

每次 val 的 per-token CSV：

```text
runs/<task>/<RUN_ID>/eval/navsim_step_*.csv
```

## Val 可视化

当前每次 val 会额外选 32 个样本保存可视化：

```text
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/index.csv
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/world_model/*.png
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/trajectory/*.png
runs/<task>/<RUN_ID>/eval/vis/step_<STEP>/bev/*.png
```

可视化样本现在取 dataset 的前 `num_samples` 个 index：

```text
idx = 0, 1, ..., num_samples - 1
```

`world_model/*.png` 左边是预测 future frame，右边是 GT future frame。

`trajectory/*.png` 是当前 `CAM_F0` 图像上的轨迹投影：

- 绿色：GT future trajectory
- 红色：predicted future trajectory

`bev/*.png` 是简单 ego-frame BEV，不画地图和 annotation：

- 绿色：GT future trajectory
- 红色：predicted future trajectory
- 黑色框：ego vehicle

`psnr_rg` 和 `ssim_rg` 只在这 32 个可视化样本上计算，不是 whole val 的图像指标。

## NavSIM Test 评估

加载训练好的 checkpoint 在 navtest 上评估：

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/navsim_v1_uncond_camf0_352x640_1e-4/eval_navtest_step_XXXXXX
```

多卡 navtest 用 `torchrun`，每张卡处理一部分 test set，rank 0 负责汇总并保存 CSV/submission/可视化：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/navsim_v1_uncond_camf0_352x640_1e-4/eval_navtest_step_XXXXXX
```

多卡评估时不要手动传 `eval_device`，脚本会自动使用 `cuda:<LOCAL_RANK>`。

输出：

```text
navsim_test_metrics.csv
submission.pkl
eval_config.yaml
vis/index.csv
vis/world_model/*.png
vis/trajectory/*.png
vis/bev/*.png
```

`navsim_test_metrics.csv` 现在是更容易读的格式：

```text
idx,token,log_name,metrics
0,<token>,<log>,"ade: ...; fde: ...; score: ..."
1,<token>,<log>,"ade: ...; fde: ...; score: ..."
-1,average,,"ade: ...; fde: ...; score: ..."
```

终端只额外输出最后的 average metrics，不逐 sample 打印。

当前 task 默认通过 `eval_visualization` 开启 32 个样本的 test 可视化。这里取的是 test dataset 的前 32 个样本。test 脚本会默认沿用这组配置；如果需要单独覆盖，可以加 `test_visualization`：

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

快速只看前 4 个 test sample：

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_vis4 \
  +test_visualization.num_samples=4
```

如果只想跑 metrics，不保存图：

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_step_XXXXXX \
  eval_visualization.enabled=false
```

如果只想关掉 BEV：

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/run/checkpoints/weights/step_XXXXXX.pt \
  output_dir=./runs/eval_navtest_step_XXXXXX \
  +test_visualization.bev=false
```

## Bad Sample 筛选

如果已经用 `scripts/evaluate_navsim.py` 保存过可视化，可以直接按 PDM 分数阈值筛 bad samples，不需要重新推理：

```bash
python scripts/filter_bad_navsim_samples.py \
  --metrics_csv runs/.../eval_navtest_step_060316/navsim_test_metrics.csv \
  --vis_index_csv runs/.../eval_navtest_step_060316/vis/index.csv \
  --metric score \
  --threshold 0.3
```

脚本会读取 test metrics CSV 和可视化索引：

```text
runs/.../eval_navtest_step_060316/navsim_test_metrics.csv
runs/.../eval_navtest_step_060316/vis/index.csv
```

`score` 是 test 输出里的 NAVSIM PDM score。脚本默认筛选 `score < threshold` 的样本。每次非 dry-run 运行都会刷新输出目录，因此多次运行不会混入旧的 bad-sample 结果。每个 bad sample 会被整理到单独文件夹：

```text
bad_samples/<token>_<metric>_<value>/
```

每个文件夹内包含已有的可视化：

```text
bev.png
camera_trajectory.png
future_rollout.png
metrics.txt
```

常用参数包括 `--comparison le`、`--max_samples`、`--output_dir`、`--dry_run`。如果在 test 输出上误传 `--metric pdm_score`，脚本会在存在 `score` 字段时自动按 `score` 处理。

`scripts/evaluate_navsim.py` 需要 `data.test.metric_cache_path` 有效，默认是：

```text
${NAVSIM_EXP_ROOT}/metric_cache_v1.1.0_navtest
```

如果 checkpoint 不在原 run 目录里，无法自动找到 `dataset_stats.json`，需要显式指定：

```bash
python scripts/evaluate_navsim.py \
  task=navsim_v1_uncond_camf0_352x640_1e-4 \
  resume=/path/to/step_XXXXXX.pt \
  data.test.pretrained_norm_stats=/path/to/dataset_stats.json \
  output_dir=./runs/eval_navtest_step_XXXXXX
```

## Metric Cache

Metric cache 是 NAVSIM 官方 PDM scoring 需要的缓存，不在 git 里。

navtrain/trainval cache 示例：

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=trainval \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_v1.1.0_navtrain \
  worker=single_machine_thread_pool \
  worker.max_workers=32
```

navtest cache 示例：

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_v1.1.0_navtest \
  worker=single_machine_thread_pool \
  worker.max_workers=32
```

`worker.max_workers` 可以根据机器 CPU/内存调整。

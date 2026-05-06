import csv
import os
import pickle
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.navsim_visualization import (
    save_bev_trajectory_overlay,
    save_camera_trajectory_overlay,
    save_world_model_future_sheet,
)
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)

register_default_resolvers()


def _init_distributed() -> tuple[int, int, int, bool, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    initialized_here = False
    if distributed and not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend)
        initialized_here = True
    return rank, world_size, local_rank, distributed, initialized_here


def _distributed_barrier(*, distributed: bool, local_rank: int) -> None:
    if not distributed:
        return
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[local_rank])
    else:
        torch.distributed.barrier()


def _destroy_distributed(*, distributed: bool, initialized_here: bool) -> None:
    if distributed and initialized_here and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _gather_object(obj: Any, *, rank: int, world_size: int, distributed: bool) -> Optional[list[Any]]:
    if not distributed:
        return [obj]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(obj, gathered, dst=0)
    return gathered


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_path = cfg.get("eval_checkpoint_path", None) or cfg.get("resume", None)
    if checkpoint_path in (None, "", "null"):
        raise ValueError("Set `resume=/path/to/checkpoint.pt` or `+eval_checkpoint_path=/path/to/checkpoint.pt`.")
    checkpoint_path = Path(str(checkpoint_path)).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _infer_dataset_stats_path(cfg: DictConfig, checkpoint_path: Path) -> Optional[str]:
    explicit = cfg.data.test.get("pretrained_norm_stats", None)
    if explicit not in (None, "", "null"):
        return str(explicit)

    train_stats = cfg.data.train.get("pretrained_norm_stats", None)
    if train_stats not in (None, "", "null"):
        return str(train_stats)

    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return str(candidate)
    return None


def _write_rows_csv(rows: list[dict[str, Any]], csv_path: Path, preferred: Optional[list[str]] = None) -> str:
    if preferred is None:
        preferred = [
            "idx",
            "token",
            "log_name",
            "traj_l1",
            "ade",
            "fde",
            "ade_2s",
            "fde_2s",
            "heading_mae",
            "score",
        ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = []
    for key in preferred:
        if key in keys:
            fieldnames.append(key)
            keys.remove(key)
    fieldnames.extend(sorted(keys))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


def _numeric_metric_items(row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if key in {"idx", "token", "log_name"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metrics[key] = float(value)
    return metrics


def _average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key not in {"idx", "token", "log_name"}
        and not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
    })
    average: dict[str, float] = {}
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row
            and not isinstance(row[key], bool)
            and isinstance(row[key], (int, float, np.integer, np.floating))
        ]
        if values:
            average[key] = float(np.mean(values))
    return average


def _format_metrics(metrics: dict[str, float]) -> str:
    return "; ".join(f"{key}: {value:.6g}" for key, value in sorted(metrics.items()))


def _write_readable_metrics_csv(rows: list[dict[str, Any]], csv_path: Path) -> tuple[str, dict[str, float]]:
    average = _average_metrics(rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "token", "log_name", "metrics"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "idx": row.get("idx", ""),
                    "token": row.get("token", ""),
                    "log_name": row.get("log_name", ""),
                    "metrics": _format_metrics(_numeric_metric_items(row)),
                }
            )
        writer.writerow(
            {
                "idx": -1,
                "token": "average",
                "log_name": "",
                "metrics": _format_metrics(average),
            }
        )
    return str(csv_path), average


def _visualization_cfg(cfg: DictConfig) -> dict[str, Any]:
    base_cfg = cfg.get("eval_visualization", None)
    override_cfg = cfg.get("test_visualization", None)
    if base_cfg is None and override_cfg is None:
        return {
            "enabled": False,
            "num_samples": 0,
            "world_model": True,
            "trajectory": True,
            "bev": True,
        }

    def get_value(key: str, default: Any) -> Any:
        if override_cfg is not None and key in override_cfg:
            return override_cfg.get(key)
        if base_cfg is not None and key in base_cfg:
            return base_cfg.get(key)
        return default

    return {
        "enabled": bool(get_value("enabled", False)),
        "num_samples": int(get_value("num_samples", 32)),
        "world_model": bool(get_value("world_model", True)),
        "trajectory": bool(get_value("trajectory", True)),
        "bev": bool(get_value("bev", True)),
    }


def _select_first_indices(dataset_len: int, num_samples: int) -> set[int]:
    dataset_len = int(dataset_len)
    num_samples = min(max(int(num_samples), 0), dataset_len)
    if num_samples <= 0:
        return set()
    return set(range(num_samples))


def _safe_token_for_filename(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(token))


def _write_visualization_csv(rows: list[dict[str, Any]], vis_dir: Path) -> str:
    vis_dir.mkdir(parents=True, exist_ok=True)
    preferred = [
        "idx",
        "token",
        "log_name",
        "psnr_rg",
        "ssim_rg",
        "world_model_path",
        "trajectory_path",
        "bev_path",
    ]
    return _write_rows_csv(rows, vis_dir / "index.csv", preferred)


def _save_navsim_test_visualization(
    *,
    idx: int,
    raw_sample: dict[str, Any],
    sample: dict[str, Any],
    pred_action_denorm: torch.Tensor,
    gt_action_denorm: Optional[torch.Tensor],
    model,
    test_dataset,
    output_dir: Path,
    vis_cfg: dict[str, Any],
    num_inference_steps: int,
) -> dict[str, Any]:
    token = raw_sample.get("token", str(idx))
    safe_token = _safe_token_for_filename(token)
    vis_dir = output_dir / "vis"
    row: dict[str, Any] = {
        "idx": int(idx),
        "token": token,
        "log_name": raw_sample.get("log_name", ""),
    }

    video0 = sample["video"][0]
    action = sample["action"][0] if sample["action"] is not None else None
    proprio = sample["proprio"][0, 0] if sample["proprio"] is not None else None
    input_image = video0[:, 0].unsqueeze(0)
    _, num_frames, _, _ = video0.shape

    if bool(vis_cfg.get("world_model", True)):
        pred = model.infer(
            prompt=None,
            input_image=input_image,
            num_frames=int(num_frames),
            action=action,
            action_horizon=sample["action_horizon"],
            proprio=proprio,
            context=sample["context"][0] if sample["context"] is not None else None,
            context_mask=sample["context_mask"][0] if sample["context_mask"] is not None else None,
            text_cfg_scale=1.0,
            action_cfg_scale=1.0,
            num_inference_steps=int(num_inference_steps),
            seed=42,
            rand_device="cpu",
            tiled=False,
        )
        pred_video_tensor = pil_frames_to_video_tensor(pred["video"])
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        if pred_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError(
                "NavSIM test visualization prediction/GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} gt={tuple(gt_video_tensor.shape)}"
            )

        row["psnr_rg"] = float(video_psnr(pred=pred_video_tensor[:, 1:], target=gt_video_tensor[:, 1:]))
        row["ssim_rg"] = float(video_ssim(pred=pred_video_tensor[:, 1:], target=gt_video_tensor[:, 1:]))
        world_path = vis_dir / "world_model" / f"{idx:06d}_{safe_token}.png"
        row["world_model_path"] = save_world_model_future_sheet(
            pred_video=pred_video_tensor,
            gt_video=gt_video_tensor,
            output_path=world_path,
        )

    if bool(vis_cfg.get("trajectory", True)):
        if gt_action_denorm is None:
            logger.warning("Skipping trajectory visualization for idx=%d token=%s because GT action is unavailable.", idx, token)
        else:
            if not hasattr(test_dataset, "get_visualization_data"):
                raise TypeError("NavSIM trajectory visualization requires dataset.get_visualization_data().")
            visualization_data = test_dataset.get_visualization_data(idx)
            trajectory_path = vis_dir / "trajectory" / f"{idx:06d}_{safe_token}.png"
            row["trajectory_path"] = save_camera_trajectory_overlay(
                image=visualization_data["image"],
                camera=visualization_data["camera"],
                pred_trajectory=pred_action_denorm,
                gt_trajectory=gt_action_denorm,
                output_path=trajectory_path,
            )

    if bool(vis_cfg.get("bev", True)):
        if gt_action_denorm is None:
            logger.warning("Skipping BEV visualization for idx=%d token=%s because GT action is unavailable.", idx, token)
        else:
            bev_path = vis_dir / "bev" / f"{idx:06d}_{safe_token}.png"
            row["bev_path"] = save_bev_trajectory_overlay(
                pred_trajectory=pred_action_denorm,
                gt_trajectory=gt_action_denorm,
                output_path=bev_path,
            )

    return row


def _create_pdm_objects(metric_cache_path: Optional[str]):
    if metric_cache_path in (None, "", "null"):
        raise ValueError("NavSIM test evaluation requires `data.test.metric_cache_path` for PDM metrics.")
    metric_cache_root = Path(str(metric_cache_path)).expanduser()
    if not metric_cache_root.exists():
        raise FileNotFoundError(f"Metric cache not found: {metric_cache_root}")

    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    return {
        "loader": MetricCacheLoader(metric_cache_root),
        "simulator": PDMSimulator(sampling),
        "scorer": PDMScorer(sampling),
        "sampling": sampling,
    }


def _compute_pdm_metrics(pdm_objects: Optional[dict[str, Any]], pred_trajectory: torch.Tensor, token: str) -> dict[str, float]:
    if pdm_objects is None:
        return {}
    loader = pdm_objects["loader"]
    if token not in loader.tokens:
        return {}

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    result = pdm_score(
        metric_cache=loader.get_from_token(token),
        model_trajectory=Trajectory(pred_trajectory.detach().cpu().numpy().astype(np.float32)),
        future_sampling=pdm_objects["sampling"],
        simulator=pdm_objects["simulator"],
        scorer=pdm_objects["scorer"],
    )
    return {key: float(value) for key, value in asdict(result).items()}


@torch.no_grad()
def run_navsim_evaluation(cfg: DictConfig) -> dict[str, Any]:
    rank, world_size, local_rank, distributed, initialized_here = _init_distributed()
    is_main_process = rank == 0
    setup_logging()
    output_dir = Path(str(cfg.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    if is_main_process:
        OmegaConf.save(config=cfg, f=output_dir / "eval_config.yaml")

    checkpoint_path = _resolve_checkpoint_path(cfg)
    stats_path = _infer_dataset_stats_path(cfg, checkpoint_path)
    if stats_path is None:
        raise ValueError(
            "Could not infer dataset_stats.json. Pass `data.test.pretrained_norm_stats=/path/to/dataset_stats.json`."
        )

    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    explicit_device = cfg.get("eval_device", None)
    if distributed and explicit_device in (None, "", "null") and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = str(explicit_device or _resolve_train_device())
    logger.info("Rank %d/%d loading model on %s from %s", rank, world_size, device, checkpoint_path)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(checkpoint_path), optimizer=None)
    model.eval()

    if is_main_process:
        logger.info("Building NavSIM test dataset with pretrained_norm_stats=%s", stats_path)
    test_dataset = instantiate(cfg.data.test, pretrained_norm_stats=stats_path)
    pdm_objects = _create_pdm_objects(test_dataset.metric_cache_path)
    vis_cfg = _visualization_cfg(cfg)
    vis_indices = (
        _select_first_indices(len(test_dataset), int(vis_cfg["num_samples"]))
        if bool(vis_cfg.get("enabled", False))
        else set()
    )
    if vis_indices and is_main_process:
        logger.info("Saving NavSIM test visualizations for %d samples under %s", len(vis_indices), output_dir / "vis")

    rows: list[dict[str, Any]] = []
    vis_rows: list[dict[str, Any]] = []
    predictions = {}
    local_indices = range(rank, len(test_dataset), world_size)
    for idx in tqdm(local_indices, desc=f"Evaluate NavSIM test rank {rank}", disable=not is_main_process):
        raw_sample = test_dataset[idx]
        sample = Wan22Trainer._to_batched_eval_sample(raw_sample)
        video0 = sample["video"][0]
        action = sample["action"][0] if sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if sample["proprio"] is not None else None
        input_image = video0[:, 0].unsqueeze(0)

        pred = model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=sample["action_horizon"],
            proprio=proprio,
            context=sample["context"][0] if sample["context"] is not None else None,
            context_mask=sample["context_mask"][0] if sample["context_mask"] is not None else None,
            num_inference_steps=int(cfg.eval_num_inference_steps),
            seed=42,
            rand_device="cpu",
            tiled=False,
        )
        pred_action_denorm = test_dataset.denormalize_action(pred["action"])
        token = raw_sample.get("token", str(idx))
        row = {
            "idx": int(idx),
            "token": token,
            "log_name": raw_sample.get("log_name", ""),
        }
        gt_action_denorm = None
        if action is not None:
            gt_action_denorm = test_dataset.denormalize_action(action.detach().cpu())
            row.update(Wan22Trainer._navsim_trajectory_metrics(pred_action_denorm, gt_action_denorm))
        row.update(_compute_pdm_metrics(pdm_objects, pred_action_denorm, token))
        rows.append(row)

        if idx in vis_indices:
            vis_rows.append(
                _save_navsim_test_visualization(
                    idx=idx,
                    raw_sample=raw_sample,
                    sample=sample,
                    pred_action_denorm=pred_action_denorm,
                    gt_action_denorm=gt_action_denorm,
                    model=model,
                    test_dataset=test_dataset,
                    output_dir=output_dir,
                    vis_cfg=vis_cfg,
                    num_inference_steps=int(cfg.eval_num_inference_steps),
                )
            )

        from navsim.common.dataclasses import Trajectory

        predictions[token] = Trajectory(pred_action_denorm.detach().cpu().numpy().astype(np.float32))

    gathered_rows = _gather_object(rows, rank=rank, world_size=world_size, distributed=distributed)
    gathered_vis_rows = _gather_object(vis_rows, rank=rank, world_size=world_size, distributed=distributed)
    gathered_predictions = _gather_object(predictions, rank=rank, world_size=world_size, distributed=distributed)
    if not is_main_process:
        _distributed_barrier(distributed=distributed, local_rank=local_rank)
        _destroy_distributed(distributed=distributed, initialized_here=initialized_here)
        return {"num_samples": len(rows)}

    rows = []
    for part in gathered_rows or []:
        if part:
            rows.extend(part)
    vis_rows = []
    for part in gathered_vis_rows or []:
        if part:
            vis_rows.extend(part)
    predictions = {}
    for part in gathered_predictions or []:
        if part:
            predictions.update(part)

    rows = sorted(rows, key=lambda row: int(row.get("idx", 0)))
    csv_path, average = _write_readable_metrics_csv(rows, output_dir / "navsim_test_metrics.csv")
    submission_path = output_dir / "submission.pkl"
    with open(submission_path, "wb") as f:
        pickle.dump(
            {
                "team_name": "FastWAM-NavSIM",
                "authors": "",
                "email": "",
                "institution": "",
                "country / region": "",
                "predictions": [predictions],
            },
            f,
        )
    logger.info("Saved NavSIM test metrics to %s", csv_path)
    if average:
        logger.info("NavSIM test average metrics: %s", _format_metrics(average))
    logger.info("Saved NavSIM submission pickle to %s", submission_path)
    result = {"csv_path": csv_path, "submission_path": str(submission_path), "num_samples": len(rows)}
    if vis_rows:
        vis_rows = sorted(vis_rows, key=lambda row: int(row.get("idx", 0)))
        vis_csv_path = _write_visualization_csv(vis_rows, output_dir / "vis")
        logger.info("Saved NavSIM test visualization index to %s", vis_csv_path)
        result["visualization_csv_path"] = vis_csv_path
        result["image_num_samples"] = len(vis_rows)
    _distributed_barrier(distributed=distributed, local_rank=local_rank)
    _destroy_distributed(distributed=distributed, initialized_here=initialized_here)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_navsim_evaluation(cfg)


if __name__ == "__main__":
    main()

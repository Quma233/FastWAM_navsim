import logging
import csv
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.navsim_visualization import (
    save_camera_trajectory_overlay,
    save_world_model_future_sheet,
)
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.sanitize_nonfinite_gradients = bool(cfg.get("sanitize_nonfinite_gradients", False))
        self.check_finite_parameters = bool(cfg.get("check_finite_parameters", False))
        self.optimizer_impl = str(cfg.get("optimizer_impl", "adamw")).strip().lower()
        self.optimizer_foreach = cfg.get("optimizer_foreach", None)
        self.optimizer_fused = cfg.get("optimizer_fused", None)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f sanitize_nonfinite_gradients=%s",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            self.sanitize_nonfinite_gradients,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.95),
        }
        if self.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = bool(self.optimizer_foreach)
        if self.optimizer_fused is not None:
            optimizer_kwargs["fused"] = bool(self.optimizer_fused)
        if self.optimizer_impl == "adamw":
            self.optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)
        elif self.optimizer_impl == "deepspeed_fused_adam":
            from deepspeed.ops.adam import FusedAdam

            self.optimizer = FusedAdam(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
                adam_w_mode=True,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer_impl={self.optimizer_impl!r}; "
                "expected 'adamw' or 'deepspeed_fused_adam'."
            )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _sanitize_nonfinite_gradients(self) -> int:
        if not self.sanitize_nonfinite_gradients:
            return 0

        local_count = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
        for param in self.model.parameters():
            grad = param.grad
            if grad is None or not grad.is_floating_point():
                continue
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all().item()):
                continue
            nonfinite_count = (~finite_mask).sum()
            local_count += nonfinite_count.to(device=local_count.device, dtype=local_count.dtype)
            torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)

        total_count = self.accelerator.reduce(local_count, reduction="sum")
        return int(total_count.item())

    def _find_nonfinite_trainable_parameter(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            tensor = param.detach()
            if bool(torch.isfinite(tensor).all().item()):
                continue
            count = int((~torch.isfinite(tensor)).sum().item())
            return name, count, tuple(tensor.shape), str(tensor.dtype)
        return None

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @staticmethod
    def _angle_abs_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.atan2(torch.sin(diff), torch.cos(diff)).abs()

    @classmethod
    def _navsim_trajectory_metrics(cls, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred = pred.detach().cpu().float()
        target = target.detach().cpu().float()
        if pred.shape != target.shape:
            raise ValueError(f"NavSIM trajectory shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError(f"Expected NavSIM trajectory [T,3], got {tuple(pred.shape)}")
        pos_dist = torch.linalg.norm(pred[:, :2] - target[:, :2], dim=-1)
        horizon_2s = min(4, pos_dist.shape[0])
        heading_mae = cls._angle_abs_diff(pred[:, 2], target[:, 2]).mean()
        return {
            "traj_l1": float((pred - target).abs().mean().item()),
            "ade": float(pos_dist.mean().item()),
            "fde": float(pos_dist[-1].item()),
            "ade_2s": float(pos_dist[:horizon_2s].mean().item()),
            "fde_2s": float(pos_dist[horizon_2s - 1].item()),
            "heading_mae": float(heading_mae.item()),
        }

    def _gather_object_rows(self, local_rows: list[dict]):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(self.accelerator.num_processes)] if self.accelerator.is_main_process else None
            torch.distributed.gather_object(local_rows, gathered, dst=0)
            if not self.accelerator.is_main_process:
                return None
            rows = []
            for part in gathered:
                if part:
                    rows.extend(part)
            return rows
        return local_rows

    def _navsim_eval_visualization_cfg(self) -> dict:
        cfg = self.cfg.get("eval_visualization", None)
        if cfg is None:
            return {
                "enabled": False,
                "num_samples": 0,
                "world_model": True,
                "trajectory": True,
            }
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "num_samples": int(cfg.get("num_samples", 32)),
            "world_model": bool(cfg.get("world_model", True)),
            "trajectory": bool(cfg.get("trajectory", True)),
        }

    @staticmethod
    def _select_evenly_spaced_indices(dataset_len: int, num_samples: int) -> set[int]:
        dataset_len = int(dataset_len)
        num_samples = min(max(int(num_samples), 0), dataset_len)
        if num_samples <= 0:
            return set()
        if num_samples >= dataset_len:
            return set(range(dataset_len))
        indices = np.linspace(0, dataset_len - 1, num=num_samples, dtype=np.int64).tolist()
        return {int(idx) for idx in indices}

    @staticmethod
    def _safe_token_for_filename(token: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(token))

    def _write_rows_csv(self, rows: list[dict], csv_path: str, preferred: list[str]) -> str:
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
        return csv_path

    def _write_navsim_eval_csv(self, rows: list[dict]) -> str:
        csv_path = os.path.join(self.eval_dir, f"navsim_step_{self.global_step:06d}.csv")
        preferred = [
            "idx",
            "token",
            "log_name",
            "val_loss",
            "traj_l1",
            "ade",
            "fde",
            "ade_2s",
            "fde_2s",
            "heading_mae",
            "pdm_score",
        ]
        return self._write_rows_csv(rows, csv_path, preferred)

    def _write_navsim_visualization_csv(self, rows: list[dict], vis_dir: str) -> str:
        csv_path = os.path.join(vis_dir, "index.csv")
        preferred = [
            "idx",
            "token",
            "log_name",
            "psnr_rg",
            "ssim_rg",
            "world_model_path",
            "trajectory_path",
        ]
        return self._write_rows_csv(rows, csv_path, preferred)

    def _save_navsim_eval_visualization(
        self,
        *,
        idx: int,
        raw_sample: dict,
        sample: dict,
        pred_action_denorm: torch.Tensor,
        gt_action_denorm: torch.Tensor,
        model,
        vis_cfg: dict,
    ) -> dict:
        token = raw_sample.get("token", str(idx))
        safe_token = self._safe_token_for_filename(token)
        vis_dir = os.path.join(self.eval_dir, "vis", f"step_{self.global_step:06d}")
        row = {
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
            infer_kwargs = {
                "prompt": None,
                "input_image": input_image,
                "num_frames": int(num_frames),
                "action": action,
                "action_horizon": sample["action_horizon"],
                "proprio": proprio,
                "context": sample["context"][0] if sample["context"] is not None else None,
                "context_mask": sample["context_mask"][0] if sample["context_mask"] is not None else None,
                "text_cfg_scale": 1.0,
                "action_cfg_scale": 1.0,
                "num_inference_steps": self.eval_num_inference_steps,
                "seed": 42,
                "rand_device": "cpu",
                "tiled": False,
            }
            pred = model.infer(**infer_kwargs)
            pred_video_tensor = pil_frames_to_video_tensor(pred["video"])
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            if pred_video_tensor.shape != gt_video_tensor.shape:
                raise ValueError(
                    "NavSIM visualization prediction/GT shape mismatch: "
                    f"pred={tuple(pred_video_tensor.shape)} gt={tuple(gt_video_tensor.shape)}"
                )

            row["psnr_rg"] = float(video_psnr(pred=pred_video_tensor[:, 1:], target=gt_video_tensor[:, 1:]))
            row["ssim_rg"] = float(video_ssim(pred=pred_video_tensor[:, 1:], target=gt_video_tensor[:, 1:]))
            world_path = os.path.join(vis_dir, "world_model", f"{idx:06d}_{safe_token}.png")
            row["world_model_path"] = save_world_model_future_sheet(
                pred_video=pred_video_tensor,
                gt_video=gt_video_tensor,
                output_path=world_path,
            )

        if bool(vis_cfg.get("trajectory", True)):
            if not hasattr(self.val_dataset, "get_visualization_data"):
                raise TypeError("NavSIM trajectory visualization requires dataset.get_visualization_data().")
            visualization_data = self.val_dataset.get_visualization_data(idx)
            trajectory_path = os.path.join(vis_dir, "trajectory", f"{idx:06d}_{safe_token}.png")
            row["trajectory_path"] = save_camera_trajectory_overlay(
                image=visualization_data["image"],
                camera=visualization_data["camera"],
                pred_trajectory=pred_action_denorm,
                gt_trajectory=gt_action_denorm,
                output_path=trajectory_path,
            )

        return row

    @torch.no_grad()
    def evaluate_full_dataset(self):
        if self.val_dataset is None:
            return None
        if not hasattr(self.val_dataset, "denormalize_action"):
            raise TypeError("eval_full_dataset=true requires a validation dataset with denormalize_action().")

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        local_rows: list[dict] = []
        local_vis_rows: list[dict] = []
        vis_cfg = self._navsim_eval_visualization_cfg()
        vis_indices = (
            self._select_evenly_spaced_indices(len(self.val_dataset), int(vis_cfg["num_samples"]))
            if bool(vis_cfg["enabled"])
            else set()
        )
        indices = range(self.accelerator.process_index, len(self.val_dataset), self.accelerator.num_processes)
        for idx in indices:
            raw_sample = self.val_dataset[idx]
            sample = self._to_batched_eval_sample(raw_sample)

            with self.accelerator.autocast():
                val_loss, _ = model.training_loss(sample)
            val_loss_value = float(val_loss.detach().float().item())

            video0 = sample["video"][0]
            action = sample["action"][0] if sample["action"] is not None else None
            if action is None:
                raise ValueError("NavSIM full validation requires `action` in samples.")
            proprio = sample["proprio"][0, 0] if sample["proprio"] is not None else None
            input_image = video0[:, 0].unsqueeze(0)

            infer_kwargs = {
                "prompt": None,
                "input_image": input_image,
                "action_horizon": sample["action_horizon"],
                "proprio": proprio,
                "context": sample["context"][0] if sample["context"] is not None else None,
                "context_mask": sample["context_mask"][0] if sample["context_mask"] is not None else None,
                "num_inference_steps": self.eval_num_inference_steps,
                "seed": 42,
                "rand_device": "cpu",
                "tiled": False,
            }
            pred = model.infer_action(**infer_kwargs)
            pred_action = pred["action"]

            pred_action_denorm = self.val_dataset.denormalize_action(pred_action)
            gt_action_denorm = self.val_dataset.denormalize_action(action.detach().cpu())
            traj_metrics = self._navsim_trajectory_metrics(pred_action_denorm, gt_action_denorm)

            token = raw_sample.get("token", str(idx))
            row = {
                "idx": int(idx),
                "token": token,
                "log_name": raw_sample.get("log_name", ""),
                "val_loss": val_loss_value,
                **traj_metrics,
            }
            if hasattr(self.val_dataset, "compute_pdm_metrics"):
                pdm_metrics = self.val_dataset.compute_pdm_metrics(pred_action_denorm, token)
                if pdm_metrics is not None:
                    row.update(pdm_metrics)
            local_rows.append(row)
            if idx in vis_indices:
                local_vis_rows.append(
                    self._save_navsim_eval_visualization(
                        idx=idx,
                        raw_sample=raw_sample,
                        sample=sample,
                        pred_action_denorm=pred_action_denorm,
                        gt_action_denorm=gt_action_denorm,
                        model=model,
                        vis_cfg=vis_cfg,
                    )
                )

        if was_dit_training:
            self._set_dit_only_train_mode()

        all_rows = self._gather_object_rows(local_rows)
        all_vis_rows = self._gather_object_rows(local_vis_rows)
        if not self.accelerator.is_main_process:
            return None
        if not all_rows:
            return None

        all_rows = sorted(all_rows, key=lambda row: int(row.get("idx", 0)))
        csv_path = self._write_navsim_eval_csv(all_rows)
        numeric_keys = sorted({
            key
            for row in all_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "idx"
        })
        result = {
            "eval_mode": "navsim_full",
            "num_samples": int(len(all_rows)),
            "csv_path": csv_path,
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in all_rows if key in row and isinstance(row[key], (int, float))]
            if values:
                result[key] = float(np.mean(values))
        if all_vis_rows:
            all_vis_rows = sorted(all_vis_rows, key=lambda row: int(row.get("idx", 0)))
            vis_dir = os.path.join(self.eval_dir, "vis", f"step_{self.global_step:06d}")
            vis_csv_path = self._write_navsim_visualization_csv(all_vis_rows, vis_dir)
            result["visualization_csv_path"] = vis_csv_path
            result["image_num_samples"] = int(len(all_vis_rows))
            for key in ("psnr_rg", "ssim_rg"):
                values = [
                    float(row[key])
                    for row in all_vis_rows
                    if key in row and isinstance(row[key], (int, float))
                ]
                if values:
                    result[key] = float(np.mean(values))
            logger.info("Saved NAVSIM validation visualizations to %s", vis_dir)
        logger.info("Saved NAVSIM full validation CSV to %s", csv_path)
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        if bool(self.cfg.get("eval_full_dataset", False)):
            return self.evaluate_full_dataset()

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                if not bool(torch.isfinite(loss.detach()).all().item()):
                    raise RuntimeError(f"Non-finite training loss at step {self.global_step}: {float(loss.detach().float().item())}")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    sanitized_gradients = self._sanitize_nonfinite_gradients()
                    if sanitized_gradients > 0 and self.accelerator.is_main_process:
                        logger.warning(
                            "Sanitized %d non-finite gradient values before clipping at step %d.",
                            sanitized_gradients,
                            self.global_step + 1,
                        )
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    grad_norm_tensor_for_check = torch.as_tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    if not bool(torch.isfinite(grad_norm_tensor_for_check).all().item()):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(f"Non-finite gradient norm after sanitization at step {self.global_step + 1}: {float(grad_norm_tensor_for_check.item())}")
                    self.optimizer.step()
                    if self.check_finite_parameters:
                        bad_param = self._find_nonfinite_trainable_parameter()
                        if bad_param is not None:
                            name, count, shape, dtype = bad_param
                            raise RuntimeError(
                                f"Non-finite trainable parameter after optimizer step {self.global_step + 1}: "
                                f"name={name} count={count} shape={shape} dtype={dtype}"
                            )
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e grad_norm=%.4f speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            global_grad_norm,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/sanitized_gradients": sanitized_gradients,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            if metrics.get("eval_mode") == "navsim_full":
                                description = (
                                    "[eval] step=%d samples=%d val_loss=%.4f ADE=%.4f FDE=%.4f traj_l1=%.4f heading_mae=%.4f"
                                    % (
                                        self.global_step,
                                        int(metrics.get("num_samples", 0)),
                                        metrics["val_loss"],
                                        metrics["ade"],
                                        metrics["fde"],
                                        metrics["traj_l1"],
                                        metrics["heading_mae"],
                                    )
                                )
                                if "pdm_score" in metrics:
                                    description += " pdm_score=%.4f" % metrics["pdm_score"]
                                if "psnr_rg" in metrics and "ssim_rg" in metrics:
                                    description += " image_samples=%d psnr_rg=%.4f ssim_rg=%.4f" % (
                                        int(metrics.get("image_num_samples", 0)),
                                        metrics["psnr_rg"],
                                        metrics["ssim_rg"],
                                    )
                                if "csv_path" in metrics:
                                    description += " csv=%s" % metrics["csv_path"]
                                if "visualization_csv_path" in metrics:
                                    description += " vis=%s" % metrics["visualization_csv_path"]
                                logger.info(description)
                                eval_payload = {}
                                for key, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        eval_payload[f"eval/{key}"] = float(value)
                                self._wandb_log(eval_payload)
                            else:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                logger.info(description)
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/psnr_rg": float(metrics["psnr_rg"]),
                                    "eval/ssim_rg": float(metrics["ssim_rg"]),
                                    "eval/psnr_rd": float(metrics["psnr_rd"]),
                                    "eval/ssim_rd": float(metrics["ssim_rd"]),
                                    "eval/psnr_dg": float(metrics["psnr_dg"]),
                                    "eval/ssim_dg": float(metrics["ssim_dg"]),
                                }
                                if "action_l2" in metrics:
                                    eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                                if "action_l1" in metrics:
                                    eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                                self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        

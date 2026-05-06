import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from accelerate import PartialState
from fastwam.datasets.dataset_utils import Normalize
from fastwam.datasets.lerobot.utils.normalizer import (
    LinearNormalizer,
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

NAVSIM_V1_PROMPT = "A front camera video from an autonomous vehicle. Predict the ego vehicle's 4-second future trajectory."
NAVSIM_INTERVAL_LENGTH = 0.5


class NavsimV1FastWAMDataset(Dataset):
    """NAVSIM v1 adapter that emits FastWAM training samples.

    The dataset intentionally keeps the model-facing contract identical to the
    existing FastWAM datasets: video in [-1, 1], normalized action/proprio, and
    cached Wan text context.
    """

    def __init__(
        self,
        navsim_log_path: str,
        sensor_blobs_path: str,
        shape_meta: Optional[dict[str, Any]] = None,
        split: str = "train",
        split_proportion: float = 0.95,
        split_mode: str = "official",
        navsim_split_name: Optional[str] = None,
        split_config_root: Optional[str] = None,
        log_split_file: str = "default_train_val_test_log_split.yaml",
        num_future_frames: int = 8,
        frame_interval: int = 1,
        has_route: bool = True,
        max_scenes: Optional[int] = None,
        camera: str = "CAM_F0",
        video_size: list[int] | tuple[int, int] = (352, 640),
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        prompt: str = NAVSIM_V1_PROMPT,
        pretrained_norm_stats: Optional[str] = None,
        is_training_set: bool = False,
        processor: Optional[Any] = None,
        use_stepwise_action_norm: bool = False,
        norm_default_mode: str = "z-score",
        norm_exception_mode: Optional[dict[str, dict[str, str]]] = None,
        metric_cache_path: Optional[str] = None,
        filter_missing_images: bool = True,
    ):
        super().__init__()
        self.navsim_log_path = Path(str(navsim_log_path)).expanduser()
        self.sensor_blobs_path = Path(str(sensor_blobs_path)).expanduser()
        self.split = str(split).lower().strip()
        self.split_proportion = float(split_proportion)
        self.split_mode = str(split_mode).lower().strip()
        self.navsim_split_name = (
            str(navsim_split_name).strip()
            if navsim_split_name not in (None, "", "null")
            else ("navtest" if self.split == "test" else "navtrain")
        )
        self.split_config_root = self._resolve_split_config_root(split_config_root)
        self.log_split_file = str(log_split_file)
        self.num_future_frames = int(num_future_frames)
        self.num_video_frames = self.num_future_frames + 1
        self.frame_interval = int(frame_interval)
        self.has_route = bool(has_route)
        self.max_scenes = None if max_scenes is None else int(max_scenes)
        self.camera = str(camera).upper()
        self.camera_attr = self.camera.lower()
        self.video_size = [int(video_size[0]), int(video_size[1])]
        self.text_embedding_cache_dir = None if text_embedding_cache_dir is None else str(text_embedding_cache_dir)
        self.context_len = int(context_len)
        self.prompt = str(prompt)
        self.is_training_set = bool(is_training_set)
        self.use_stepwise_action_norm = bool(use_stepwise_action_norm)
        self.norm_default_mode = str(norm_default_mode)
        self.norm_exception_mode = norm_exception_mode
        self.metric_cache_path = None if metric_cache_path in (None, "", "null") else str(metric_cache_path)
        self._scene_frames_by_token: dict[str, list[dict[str, Any]]] = {}
        self._sensor_config = self._build_sensor_config()

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split={split!r}; expected train/val/test.")
        if self.split_mode not in {"official", "proportion"}:
            raise ValueError(
                f"Unsupported split_mode={split_mode!r}; expected official/proportion."
            )
        if self.num_future_frames != 8:
            raise ValueError("NavSIM v1 FastWAM adapter expects exactly 8 future frames for 4s horizon.")
        if self.num_video_frames % 4 != 1:
            raise ValueError(f"FastWAM video frames must satisfy T % 4 == 1, got {self.num_video_frames}.")
        if self.video_size[0] % 16 != 0 or self.video_size[1] % 16 != 0:
            raise ValueError(f"video_size must be multiples of 16, got {self.video_size}.")
        if not self.navsim_log_path.is_dir():
            raise FileNotFoundError(f"navsim_log_path does not exist: {self.navsim_log_path}")
        if not self.sensor_blobs_path.is_dir():
            raise FileNotFoundError(f"sensor_blobs_path does not exist: {self.sensor_blobs_path}")

        if isinstance(shape_meta, DictConfig):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.shape_meta = shape_meta or self._default_shape_meta()
        self._validate_shape_meta(self.shape_meta)

        # Keep these attributes available under data.train.processor.* for model config interpolation.
        self.processor = processor
        self.action_output_dim = int(self.shape_meta["action"][0]["shape"])
        self.proprio_output_dim = int(self.shape_meta["state"][0]["shape"])

        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
        self.samples = self._build_index(filter_missing_images=filter_missing_images)
        if not self.samples:
            raise RuntimeError(
                "No eligible NAVSIM samples found after CAM_F0/current+future filtering. "
                f"navsim_log_path={self.navsim_log_path} sensor_blobs_path={self.sensor_blobs_path}"
            )

        dataset_stats = self._load_or_compute_stats(pretrained_norm_stats)
        self.normalizer = LinearNormalizer(
            shape_meta=self.shape_meta,
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )
        self._metric_cache_loader = None
        self._metric_cache_checked = False

        logger.info(
            "NAVSIM v1 %s dataset ready: samples=%d split_mode=%s navsim_split=%s log_path=%s camera=%s video_size=%s",
            self.split,
            len(self.samples),
            self.split_mode,
            self.navsim_split_name,
            self.navsim_log_path,
            self.camera,
            self.video_size,
        )

    @staticmethod
    def _default_shape_meta() -> dict[str, Any]:
        return {
            "action": [{"key": "default", "raw_shape": 3, "shape": 3}],
            "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
        }

    @staticmethod
    def _validate_shape_meta(shape_meta: dict[str, Any]) -> None:
        action_meta = shape_meta.get("action", [])
        state_meta = shape_meta.get("state", [])
        if len(action_meta) != 1 or int(action_meta[0]["shape"]) != 3:
            raise ValueError("NavSIM action shape_meta must contain one 3D trajectory action field.")
        if len(state_meta) != 1 or int(state_meta[0]["shape"]) != 8:
            raise ValueError("NavSIM state shape_meta must contain one 8D ego status field.")

    def _build_sensor_config(self):
        from navsim.common.dataclasses import SensorConfig

        include = list(range(self.num_video_frames))
        kwargs = {
            "cam_f0": False,
            "cam_l0": False,
            "cam_l1": False,
            "cam_l2": False,
            "cam_r0": False,
            "cam_r1": False,
            "cam_r2": False,
            "cam_b0": False,
            "lidar_pc": False,
        }
        if self.camera_attr not in kwargs:
            raise ValueError(f"Unsupported NAVSIM camera={self.camera!r}.")
        kwargs[self.camera_attr] = include
        return SensorConfig(**kwargs)

    @staticmethod
    def _string_list(values: Optional[Any]) -> Optional[list[str]]:
        if values is None:
            return None
        return [str(value) for value in values]

    @staticmethod
    def _resolve_split_config_root(split_config_root: Optional[str]) -> Path:
        def candidates_from(root: Path) -> list[Path]:
            return [root, root / "navsim" / "planning" / "script" / "config"]

        candidates: list[Path] = []
        if split_config_root not in (None, "", "null"):
            candidates.extend(candidates_from(Path(str(split_config_root)).expanduser()))
        navsim_devkit_root = os.environ.get("NAVSIM_DEVKIT_ROOT")
        if navsim_devkit_root:
            candidates.extend(candidates_from(Path(navsim_devkit_root).expanduser()))
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / "third_party" / "navsim" / "navsim" / "planning" / "script" / "config")

        for candidate in candidates:
            if (candidate / "common" / "train_test_split").is_dir() and (candidate / "training").is_dir():
                return candidate
        return candidates[0]

    def _load_yaml_dict(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"NAVSIM split config not found: {path}")
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)

    def _load_train_val_test_logs(self, split: str) -> Optional[list[str]]:
        path = self.split_config_root / "training" / self.log_split_file
        cfg = self._load_yaml_dict(path)
        return self._string_list(cfg.get(f"{split}_logs"))

    def _load_scene_filter_config(self) -> dict[str, Any]:
        path = (
            self.split_config_root
            / "common"
            / "train_test_split"
            / "scene_filter"
            / f"{self.navsim_split_name}.yaml"
        )
        return self._load_yaml_dict(path)

    def _build_official_scene_filter(self):
        from navsim.common.dataclasses import SceneFilter

        filter_cfg = self._load_scene_filter_config()
        base_log_names = self._string_list(filter_cfg.get("log_names"))
        split_log_names = self._load_train_val_test_logs(self.split)
        if split_log_names is None:
            log_names = base_log_names
        elif base_log_names is None:
            log_names = split_log_names
        else:
            split_log_set = set(split_log_names)
            log_names = [log_name for log_name in base_log_names if log_name in split_log_set]

        tokens = self._string_list(filter_cfg.get("tokens"))
        scene_filter = SceneFilter(
            num_history_frames=int(filter_cfg.get("num_history_frames", 1)),
            num_future_frames=int(filter_cfg.get("num_future_frames", max(self.num_future_frames, 0))),
            frame_interval=filter_cfg.get("frame_interval", self.frame_interval),
            has_route=bool(filter_cfg.get("has_route", self.has_route)),
            max_scenes=None,
            log_names=log_names,
            tokens=tokens,
        )
        return scene_filter

    def _collect_scene_samples(
        self,
        scene_loader: Any,
        current_frame_index: int,
        filter_missing_images: bool,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        self._scene_frames_by_token = {}
        for token in scene_loader.tokens:
            source_window = scene_loader.scene_frames_dicts[token]
            window = source_window[current_frame_index : current_frame_index + self.num_video_frames]
            if len(window) != self.num_video_frames:
                continue
            if filter_missing_images and not self._window_has_camera_images(window):
                continue
            token = str(token)
            self._scene_frames_by_token[token] = window
            samples.append(
                {
                    "token": token,
                    "log_name": str(window[0]["log_name"]),
                    "timestamp": int(window[0]["timestamp"]),
                }
            )
        samples = sorted(samples, key=lambda x: (x["log_name"], x["timestamp"], x["token"]))
        if self.max_scenes is not None:
            samples = samples[: self.max_scenes]
            selected_tokens = {sample["token"] for sample in samples}
            self._scene_frames_by_token = {
                token: self._scene_frames_by_token[token] for token in selected_tokens
            }
        return samples

    def _build_index(self, filter_missing_images: bool) -> list[dict[str, Any]]:
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SceneFilter

        if not any(self.navsim_log_path.glob("*.pkl")):
            raise FileNotFoundError(f"No NAVSIM log pickle files found in {self.navsim_log_path}")

        if self.split_mode == "official":
            scene_filter = self._build_official_scene_filter()
            scene_loader = SceneLoader(
                data_path=self.navsim_log_path,
                sensor_blobs_path=self.sensor_blobs_path,
                scene_filter=scene_filter,
                sensor_config=self._sensor_config,
            )
            return self._collect_scene_samples(
                scene_loader=scene_loader,
                current_frame_index=scene_filter.num_history_frames - 1,
                filter_missing_images=filter_missing_images,
            )

        scene_filter = SceneFilter(
            num_history_frames=1,
            num_future_frames=self.num_future_frames,
            frame_interval=self.frame_interval,
            has_route=self.has_route,
            max_scenes=None,
        )
        scene_loader = SceneLoader(
            data_path=self.navsim_log_path,
            sensor_blobs_path=self.sensor_blobs_path,
            scene_filter=scene_filter,
            sensor_config=self._sensor_config,
        )
        all_samples = self._collect_scene_samples(
            scene_loader=scene_loader,
            current_frame_index=0,
            filter_missing_images=filter_missing_images,
        )
        split_idx = int(len(all_samples) * self.split_proportion)
        split_idx = min(max(split_idx, 1), max(len(all_samples) - 1, 1))
        if self.split == "train":
            selected = all_samples[:split_idx]
        else:
            selected = all_samples[split_idx:]

        selected_tokens = {sample["token"] for sample in selected}
        self._scene_frames_by_token = {token: self._scene_frames_by_token[token] for token in selected_tokens}
        return selected

    def _window_has_camera_images(self, window: list[dict[str, Any]]) -> bool:
        for frame in window:
            camera_dict = frame.get("cams", {}).get(self.camera)
            if camera_dict is None:
                return False
            image_path = self.sensor_blobs_path / camera_dict["data_path"]
            if not image_path.is_file():
                return False
        return True

    def _load_window(self, sample_info: dict[str, Any]) -> list[dict[str, Any]]:
        token = str(sample_info["token"])
        return self._scene_frames_by_token[token]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_info = self.samples[int(idx)]
        window = self._load_window(sample_info)
        raw_video = self._load_video_tensor(window)
        video = self.normalize_transform(raw_video).permute(1, 0, 2, 3).contiguous()

        action_raw = torch.as_tensor(self._future_trajectory(window), dtype=torch.float32)
        proprio_raw = torch.as_tensor(self._ego_status(window[0]), dtype=torch.float32).unsqueeze(0)
        action, proprio = self._normalize_action_state(action_raw, proprio_raw)
        context, context_mask = self._get_cached_text_context(self.prompt)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": self.prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.zeros(self.num_video_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(self.num_future_frames, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(1, dtype=torch.bool),
            "token": sample_info["token"],
            "log_name": sample_info["log_name"],
        }

    def _load_camera_from_frame(self, frame: dict[str, Any]):
        from navsim.common.dataclasses import Cameras

        cameras = Cameras.from_camera_dict(
            sensor_blobs_path=self.sensor_blobs_path,
            camera_dict=frame["cams"],
            sensor_names=[self.camera_attr],
        )
        camera = getattr(cameras, self.camera_attr)
        if camera.image is None:
            raise FileNotFoundError(f"Missing {self.camera} image for token={frame.get('token')}")
        return camera

    def _preprocess_camera_image(self, image: np.ndarray) -> np.ndarray:
        # Drive-JEPA NAVSIM v1 front-camera preprocessing: crop vertical borders, resize.
        image = image[28:-28]
        image = cv2.resize(
            image,
            (self.video_size[1], self.video_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.ascontiguousarray(image)

    def _processed_camera_intrinsics(
        self,
        intrinsics: np.ndarray,
        raw_image_shape: tuple[int, int],
    ) -> np.ndarray:
        raw_height, raw_width = int(raw_image_shape[0]), int(raw_image_shape[1])
        cropped_height = raw_height - 56
        if cropped_height <= 0:
            raise ValueError(f"Unexpected NAVSIM camera image height: {raw_height}")
        scale_x = float(self.video_size[1]) / float(raw_width)
        scale_y = float(self.video_size[0]) / float(cropped_height)
        adjusted = np.asarray(intrinsics, dtype=np.float32).copy()
        adjusted[0, 0] *= scale_x
        adjusted[0, 2] *= scale_x
        adjusted[1, 1] *= scale_y
        adjusted[1, 2] = (adjusted[1, 2] - 28.0) * scale_y
        return adjusted

    def get_visualization_data(self, idx: int) -> dict[str, Any]:
        sample_info = self.samples[int(idx)]
        window = self._load_window(sample_info)
        camera = self._load_camera_from_frame(window[0])
        return {
            "token": sample_info["token"],
            "log_name": sample_info["log_name"],
            "image": self._preprocess_camera_image(camera.image),
            "camera": {
                "sensor2lidar_rotation": np.asarray(camera.sensor2lidar_rotation, dtype=np.float32),
                "sensor2lidar_translation": np.asarray(camera.sensor2lidar_translation, dtype=np.float32),
                "intrinsics": self._processed_camera_intrinsics(
                    np.asarray(camera.intrinsics, dtype=np.float32),
                    raw_image_shape=camera.image.shape[:2],
                ),
            },
        }

    def _load_video_tensor(self, window: list[dict[str, Any]]) -> torch.Tensor:
        frames = []
        for frame_idx, frame in enumerate(window):
            del frame_idx
            camera = self._load_camera_from_frame(frame)
            image = self._preprocess_camera_image(camera.image)
            frames.append(torch.from_numpy(image).permute(2, 0, 1))
        return torch.stack(frames, dim=0).to(torch.uint8)

    @staticmethod
    def _global_ego_pose(frame: dict[str, Any]) -> np.ndarray:
        from pyquaternion import Quaternion

        ego_translation = frame["ego2global_translation"]
        ego_quaternion = Quaternion(*frame["ego2global_rotation"])
        return np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )

    def _future_trajectory(self, window: list[dict[str, Any]]) -> np.ndarray:
        from nuplan.common.actor_state.state_representation import StateSE2
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
            convert_absolute_to_relative_se2_array,
        )

        global_ego_poses = [self._global_ego_pose(frame) for frame in window]
        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]),
            np.array(global_ego_poses[1:], dtype=np.float64),
        )
        return local_ego_poses.astype(np.float32)

    @staticmethod
    def _ego_status(frame: dict[str, Any]) -> np.ndarray:
        ego_dynamic_state = frame["ego_dynamic_state"]
        velocity = np.array(ego_dynamic_state[:2], dtype=np.float32)
        acceleration = np.array(ego_dynamic_state[2:], dtype=np.float32)
        driving_command = np.array(frame["driving_command"], dtype=np.float32)
        return np.concatenate([driving_command, velocity, acceleration], axis=-1).astype(np.float32)

    def _normalize_action_state(
        self,
        action_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = {
            "action": {"default": action_raw.clone()},
            "state": {"default": proprio_raw.clone()},
        }
        batch = self.normalizer.forward(batch)
        return batch["action"]["default"], batch["state"]["default"]

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if action.ndim == 2:
            action = action.unsqueeze(0)
            squeeze = True
        if action.ndim != 3 or action.shape[-1] != self.action_output_dim:
            raise ValueError(f"Expected action [T,3] or [B,T,3], got {tuple(action.shape)}")
        action_denorm = self.normalizer.normalizers["action"]["default"].backward(action.detach().cpu().float())
        return action_denorm.squeeze(0) if squeeze else action_denorm

    def _load_or_compute_stats(self, pretrained_norm_stats: Optional[str]) -> dict[str, Any]:
        if pretrained_norm_stats:
            stats = load_dataset_stats_from_json(pretrained_norm_stats)
            if PartialState().is_main_process:
                save_dataset_stats_to_json(stats, os.path.join(misc.get_work_dir(), "dataset_stats.json"))
            logger.info("Using NAVSIM dataset stats: %s", pretrained_norm_stats)
            return stats
        if not self.is_training_set:
            raise ValueError("pretrained_norm_stats must be provided for NAVSIM validation/test datasets.")

        if PartialState().is_main_process:
            logger.info("Calculating NAVSIM action/state normalization stats...")
            stats = self._compute_dataset_stats()
            save_dataset_stats_to_json(stats, os.path.join(misc.get_work_dir(), "dataset_stats.json"))
        else:
            stats = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj_list = [stats]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            stats = obj_list[0]
        return stats

    def _compute_dataset_stats(self) -> dict[str, Any]:
        action_sum = torch.zeros(self.action_output_dim, dtype=torch.float64)
        action_sumsq = torch.zeros(self.action_output_dim, dtype=torch.float64)
        action_min = torch.full((self.action_output_dim,), float("inf"), dtype=torch.float64)
        action_max = torch.full((self.action_output_dim,), float("-inf"), dtype=torch.float64)
        action_count = 0

        state_sum = torch.zeros(self.proprio_output_dim, dtype=torch.float64)
        state_sumsq = torch.zeros(self.proprio_output_dim, dtype=torch.float64)
        state_min = torch.full((self.proprio_output_dim,), float("inf"), dtype=torch.float64)
        state_max = torch.full((self.proprio_output_dim,), float("-inf"), dtype=torch.float64)
        state_count = 0

        for sample_info in self.samples:
            window = self._load_window(sample_info)
            action = torch.as_tensor(self._future_trajectory(window), dtype=torch.float64)
            state = torch.as_tensor(self._ego_status(window[0]), dtype=torch.float64).unsqueeze(0)
            action_sum += action.sum(dim=0)
            action_sumsq += action.pow(2).sum(dim=0)
            action_min = torch.minimum(action_min, action.min(dim=0).values)
            action_max = torch.maximum(action_max, action.max(dim=0).values)
            action_count += int(action.shape[0])

            state_sum += state.sum(dim=0)
            state_sumsq += state.pow(2).sum(dim=0)
            state_min = torch.minimum(state_min, state.min(dim=0).values)
            state_max = torch.maximum(state_max, state.max(dim=0).values)
            state_count += int(state.shape[0])

        def finalize(sum_v, sumsq_v, min_v, max_v, count):
            mean = sum_v / max(count, 1)
            var = (sumsq_v / max(count, 1)) - mean.pow(2)
            std = var.clamp(min=0.0).sqrt()
            return {
                "global_min": min_v.float(),
                "global_max": max_v.float(),
                "global_mean": mean.float(),
                "global_std": std.float(),
                "global_q01": min_v.float(),
                "global_q99": max_v.float(),
            }

        return {
            "action": {"default": finalize(action_sum, action_sumsq, action_min, action_max, action_count)},
            "state": {"default": finalize(state_sum, state_sumsq, state_min, state_max, state_count)},
        }

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Cached context must be [context_len,D], got {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1 or context_mask.shape[0] != self.context_len:
            raise ValueError(f"Cached mask must be [context_len], got {tuple(context_mask.shape)} in {cache_path}")
        return context, context_mask

    def _get_metric_cache_loader(self):
        if self._metric_cache_checked:
            return self._metric_cache_loader
        self._metric_cache_checked = True
        if not self.metric_cache_path:
            return None
        metric_cache_root = Path(self.metric_cache_path)
        metadata_dir = metric_cache_root / "metadata"
        if not metric_cache_root.exists() or not metadata_dir.exists():
            logger.warning("NAVSIM metric cache not found at %s; skipping PDM metrics.", metric_cache_root)
            return None
        try:
            from navsim.common.dataloader import MetricCacheLoader

            self._metric_cache_loader = MetricCacheLoader(metric_cache_root)
        except Exception as exc:
            logger.warning("Failed to initialize NAVSIM MetricCacheLoader at %s: %s", metric_cache_root, exc)
            self._metric_cache_loader = None
        return self._metric_cache_loader

    def compute_pdm_metrics(self, pred_trajectory: torch.Tensor, token: str) -> Optional[dict[str, float]]:
        loader = self._get_metric_cache_loader()
        if loader is None or token not in loader.tokens:
            return None
        try:
            from navsim.common.dataclasses import Trajectory
            from navsim.evaluate.pdm_score import pdm_score
            from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
            from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            metric_cache = loader.get_from_token(token)
            sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
            simulator = PDMSimulator(sampling)
            scorer = PDMScorer(sampling)
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(pred_trajectory.detach().cpu().numpy().astype(np.float32)),
                future_sampling=sampling,
                simulator=simulator,
                scorer=scorer,
            )
            return {f"pdm_{key}": float(value) for key, value in asdict(result).items()}
        except Exception as exc:
            logger.warning("PDM metric failed for token=%s: %s", token, exc)
            return None

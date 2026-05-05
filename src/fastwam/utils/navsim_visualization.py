from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def _tensor_frame_to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().float().cpu().clamp(0.0, 1.0)
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def save_world_model_future_sheet(
    pred_video: torch.Tensor,
    gt_video: torch.Tensor,
    output_path: str | Path,
) -> str:
    """Save one image per sample: each row is predicted future frame | GT frame."""
    if pred_video.shape != gt_video.shape:
        raise ValueError(f"Video shape mismatch: pred={tuple(pred_video.shape)} gt={tuple(gt_video.shape)}")
    if pred_video.ndim != 4 or pred_video.shape[0] != 3:
        raise ValueError(f"Expected video tensors [3,T,H,W], got {tuple(pred_video.shape)}")
    if pred_video.shape[1] <= 1:
        raise ValueError("Expected at least one future frame for world-model visualization.")

    _, num_frames, height, width = pred_video.shape
    label_height = 24
    rows = num_frames - 1
    sheet = Image.new("RGB", (width * 2, rows * (height + label_height)), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    for row, t in enumerate(range(1, num_frames)):
        y0 = row * (height + label_height)
        pred_img = _tensor_frame_to_image(pred_video[:, t])
        gt_img = _tensor_frame_to_image(gt_video[:, t])
        sheet.paste(pred_img, (0, y0 + label_height))
        sheet.paste(gt_img, (width, y0 + label_height))
        draw.text((8, y0 + 5), f"pred future frame {t}", fill=(255, 80, 80))
        draw.text((width + 8, y0 + 5), f"gt future frame {t}", fill=(80, 220, 120))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _project_lidar_points_to_image(
    points_lidar: np.ndarray,
    camera: dict[str, Any],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    from navsim.common.enums import LidarIndex
    from navsim.visualization.camera import _transform_pcs_to_images

    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    lidar_pc = np.zeros((6, points_lidar.shape[0]), dtype=np.float32)
    lidar_pc[LidarIndex.POSITION, :] = points_lidar.T
    pixels, mask = _transform_pcs_to_images(
        lidar_pc=lidar_pc,
        sensor2lidar_rotation=np.asarray(camera["sensor2lidar_rotation"], dtype=np.float32),
        sensor2lidar_translation=np.asarray(camera["sensor2lidar_translation"], dtype=np.float32),
        intrinsic=np.asarray(camera["intrinsics"], dtype=np.float32),
        img_shape=image_shape,
    )
    return pixels, mask


def _trajectory_to_lidar_points(trajectory: torch.Tensor | np.ndarray) -> np.ndarray:
    trajectory = trajectory.detach().cpu().float().numpy() if isinstance(trajectory, torch.Tensor) else np.asarray(trajectory)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(f"Expected trajectory [T,>=2], got {trajectory.shape}")
    points = np.zeros((trajectory.shape[0], 3), dtype=np.float32)
    points[:, :2] = trajectory[:, :2].astype(np.float32)
    return points


def _draw_projected_trajectory(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    valid_points = []
    for pixel, is_valid in zip(pixels, mask):
        if bool(is_valid) and np.isfinite(pixel).all():
            point = (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
            valid_points.append(point)
        else:
            valid_points.append(None)

    for start, end in zip(valid_points[:-1], valid_points[1:]):
        if start is not None and end is not None:
            draw.line([start, end], fill=color, width=width)

    radius = max(width + 1, 4)
    for point in valid_points:
        if point is None:
            continue
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def save_camera_trajectory_overlay(
    image: np.ndarray,
    camera: dict[str, Any],
    pred_trajectory: torch.Tensor | np.ndarray,
    gt_trajectory: torch.Tensor | np.ndarray,
    output_path: str | Path,
) -> str:
    """Draw predicted and GT ego trajectories on the current CAM_F0 image."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image.shape}")
    image_uint8 = image.astype(np.uint8, copy=False)
    canvas = Image.fromarray(image_uint8).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_shape = (int(image_uint8.shape[0]), int(image_uint8.shape[1]))

    gt_pixels, gt_mask = _project_lidar_points_to_image(
        _trajectory_to_lidar_points(gt_trajectory),
        camera=camera,
        image_shape=image_shape,
    )
    pred_pixels, pred_mask = _project_lidar_points_to_image(
        _trajectory_to_lidar_points(pred_trajectory),
        camera=camera,
        image_shape=image_shape,
    )

    _draw_projected_trajectory(draw, gt_pixels, gt_mask, color=(40, 220, 90), width=4)
    _draw_projected_trajectory(draw, pred_pixels, pred_mask, color=(255, 70, 70), width=4)

    draw.rectangle((6, 6, 190, 54), fill=(0, 0, 0))
    draw.text((14, 14), "GT traj", fill=(40, 220, 90))
    draw.text((14, 34), "Pred traj", fill=(255, 70, 70))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)

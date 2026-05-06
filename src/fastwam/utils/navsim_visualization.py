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


def _trajectory_xy(trajectory: torch.Tensor | np.ndarray) -> np.ndarray:
    trajectory = trajectory.detach().cpu().float().numpy() if isinstance(trajectory, torch.Tensor) else np.asarray(trajectory)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(f"Expected trajectory [T,>=2], got {trajectory.shape}")
    return trajectory[:, :2].astype(np.float32)


def save_bev_trajectory_overlay(
    pred_trajectory: torch.Tensor | np.ndarray,
    gt_trajectory: torch.Tensor | np.ndarray,
    output_path: str | Path,
    *,
    canvas_size: int = 640,
) -> str:
    """Draw a simple ego-frame BEV with predicted and GT future trajectories."""
    pred_xy = _trajectory_xy(pred_trajectory)
    gt_xy = _trajectory_xy(gt_trajectory)
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), pred_xy, gt_xy], axis=0)

    x_min = min(-5.0, float(np.nanmin(points[:, 0])) - 5.0)
    x_max = max(40.0, float(np.nanmax(points[:, 0])) + 5.0)
    lateral = max(20.0, float(np.nanmax(np.abs(points[:, 1]))) + 5.0)

    width = int(canvas_size)
    height = int(canvas_size)
    margin = 48
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    scale_x = plot_h / max(x_max - x_min, 1e-6)
    scale_y = plot_w / max(2.0 * lateral, 1e-6)

    def to_pixel(xy: np.ndarray) -> list[tuple[int, int]]:
        pixels = []
        for x, y in np.asarray(xy, dtype=np.float32):
            px = margin + (lateral - float(y)) * scale_y
            py = margin + (x_max - float(x)) * scale_x
            pixels.append((int(round(px)), int(round(py))))
        return pixels

    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    grid_color = (220, 220, 220)
    axis_color = (150, 150, 150)
    text_color = (40, 40, 40)
    x_tick_start = int(np.floor(x_min / 5.0) * 5)
    x_tick_end = int(np.ceil(x_max / 5.0) * 5)
    for x in range(x_tick_start, x_tick_end + 1, 5):
        y0 = to_pixel(np.array([[x, -lateral]], dtype=np.float32))[0]
        y1 = to_pixel(np.array([[x, lateral]], dtype=np.float32))[0]
        draw.line([y0, y1], fill=axis_color if x == 0 else grid_color, width=2 if x == 0 else 1)
        if x >= 0:
            label_point = to_pixel(np.array([[x, lateral]], dtype=np.float32))[0]
            draw.text((label_point[0] + 4, label_point[1] - 8), f"{x}m", fill=text_color)

    y_tick_start = int(np.floor(-lateral / 5.0) * 5)
    y_tick_end = int(np.ceil(lateral / 5.0) * 5)
    for y in range(y_tick_start, y_tick_end + 1, 5):
        p0 = to_pixel(np.array([[x_min, y]], dtype=np.float32))[0]
        p1 = to_pixel(np.array([[x_max, y]], dtype=np.float32))[0]
        draw.line([p0, p1], fill=axis_color if y == 0 else grid_color, width=2 if y == 0 else 1)

    def draw_traj(xy: np.ndarray, color: tuple[int, int, int]) -> None:
        path = to_pixel(np.concatenate([np.zeros((1, 2), dtype=np.float32), xy], axis=0))
        if len(path) >= 2:
            draw.line(path, fill=color, width=4)
        for point in path[1:]:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    draw_traj(gt_xy, (40, 180, 80))
    draw_traj(pred_xy, (230, 70, 70))

    ego_box = np.array(
        [
            [-1.5, 1.0],
            [3.0, 1.0],
            [3.0, -1.0],
            [-1.5, -1.0],
            [-1.5, 1.0],
        ],
        dtype=np.float32,
    )
    draw.line(to_pixel(ego_box), fill=(30, 30, 30), width=3)
    origin = to_pixel(np.zeros((1, 2), dtype=np.float32))[0]
    draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill=(30, 30, 30))
    draw.text((14, 12), "BEV ego frame", fill=text_color)
    draw.text((14, 34), "Green: GT traj", fill=(40, 180, 80))
    draw.text((14, 54), "Red: Pred traj", fill=(230, 70, 70))
    draw.text((width - 118, 12), "front", fill=text_color)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)

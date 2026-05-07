#!/usr/bin/env python3
"""Collect low-scoring NavSIM visualization samples into per-sample folders.

This script consumes the files produced by ``scripts/evaluate_navsim.py``:

  - ``navsim_test_metrics.csv``
  - ``vis/index.csv``

It filters the already-visualized samples by a metric threshold, then copies the
existing BEV / camera trajectory / world-model rollout images into one folder per
bad sample. It does not run model inference or recompute metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


VISUALIZATION_COLUMNS = {
    "bev_path": "bev",
    "trajectory_path": "camera_trajectory",
    "world_model_path": "future_rollout",
}

TEST_SCORE_METRIC = "score"
AUTO_METRIC_CANDIDATES = (TEST_SCORE_METRIC, "pdm_score")
METRIC_ALIASES = {"pdm_score": TEST_SCORE_METRIC}


def _safe_name(value: object, *, max_len: int = 140) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not name:
        name = "sample"
    if len(name) > max_len:
        name = name[:max_len].rstrip("._")
    return name


def _parse_metrics_field(value: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not value:
        return metrics
    for item in value.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, raw_value = item.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        try:
            metric_value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(metric_value):
            metrics[key] = metric_value
    return metrics


def _parse_numeric_columns(row: dict[str, str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, raw_value in row.items():
        if key in {"idx", "token", "log_name", "metrics"}:
            continue
        if raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[key] = value
    return metrics


def _row_id(row: dict[str, str]) -> str:
    token = str(row.get("token", "")).strip()
    if token and token != "average":
        return token
    idx = str(row.get("idx", "")).strip()
    return idx


def _load_metrics(metrics_csv: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("token", "")).strip()
            idx = str(row.get("idx", "")).strip()
            if token == "average" or idx == "-1":
                continue

            metrics = _parse_metrics_field(str(row.get("metrics", "")))
            metrics.update(_parse_numeric_columns(row))
            sample_id = _row_id(row)
            if not sample_id:
                continue
            rows[sample_id] = {
                "idx": idx,
                "token": token,
                "log_name": str(row.get("log_name", "")).strip(),
                "metrics": metrics,
            }
    return rows


def _load_visualization_rows(vis_index_csv: Path) -> list[dict[str, str]]:
    with open(vis_index_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _available_metrics(metric_rows: dict[str, dict[str, object]]) -> list[str]:
    available: set[str] = set()
    for row in metric_rows.values():
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            available.update(str(key) for key in metrics.keys())
    return sorted(available)


def _resolve_filter_metric(requested_metric: str, metric_rows: dict[str, dict[str, object]]) -> str:
    requested_metric = str(requested_metric).strip()
    available = set(_available_metrics(metric_rows))
    if not available:
        raise ValueError("No numeric metrics found in metrics_csv.")

    if requested_metric == "auto":
        candidates = AUTO_METRIC_CANDIDATES
    elif requested_metric in METRIC_ALIASES:
        candidates = (requested_metric, METRIC_ALIASES[requested_metric])
    else:
        candidates = (requested_metric,)

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"Metric {requested_metric!r} not found in metrics_csv. "
        f"Available metrics: {', '.join(sorted(available))}"
    )


def _metric_value(metrics: dict[str, float], metric_name: str) -> Optional[float]:
    return metrics.get(metric_name)


def _is_bad_sample(value: float, threshold: float, comparison: str) -> bool:
    if comparison == "lt":
        return value < threshold
    if comparison == "le":
        return value <= threshold
    if comparison == "gt":
        return value > threshold
    if comparison == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported comparison: {comparison}")


def _path_candidates(raw_path: str, bases: Iterable[Path]) -> list[Path]:
    raw_path = str(raw_path).strip()
    if not raw_path:
        return []
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return [path]

    candidates = [Path.cwd() / path]
    for base in bases:
        candidates.append(base / path)
    return candidates


def _resolve_existing_path(raw_path: str, bases: Iterable[Path]) -> Optional[Path]:
    for candidate in _path_candidates(raw_path, bases):
        if candidate.is_file():
            return candidate
    return None


def _copy_visualizations(
    *,
    vis_row: dict[str, str],
    sample_dir: Path,
    bases: Iterable[Path],
    dry_run: bool,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for column, output_stem in VISUALIZATION_COLUMNS.items():
        raw_path = str(vis_row.get(column, "")).strip()
        if not raw_path:
            copied[column] = ""
            continue
        source = _resolve_existing_path(raw_path, bases)
        if source is None:
            copied[column] = "missing"
            continue
        extension = source.suffix or ".png"
        destination = sample_dir / f"{output_stem}{extension}"
        copied[column] = str(destination)
        if not dry_run:
            sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return copied


def _write_sample_metrics(sample_dir: Path, row: dict[str, object], metric_name: str, metric_value: float) -> None:
    metrics = row["metrics"]
    assert isinstance(metrics, dict)
    with open(sample_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"idx: {row.get('idx', '')}\n")
        f.write(f"token: {row.get('token', '')}\n")
        f.write(f"log_name: {row.get('log_name', '')}\n")
        f.write(f"filter_metric: {metric_name}\n")
        f.write(f"filter_value: {metric_value:.10g}\n")
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)):
                f.write(f"{key}: {float(value):.10g}\n")


def _write_manifest(rows: list[dict[str, object]], output_dir: Path) -> Path:
    manifest_path = output_dir / "bad_samples.csv"
    fieldnames = [
        "idx",
        "token",
        "log_name",
        "metric",
        "value",
        "sample_dir",
        "bev_path",
        "trajectory_path",
        "world_model_path",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return manifest_path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _refresh_output_dir(output_dir: Path, *, metrics_csv: Path, vis_index_csv: Path) -> None:
    resolved = output_dir.resolve()
    protected = {
        Path("/").resolve(),
        Path.cwd().resolve(),
        metrics_csv.parent.resolve(),
        vis_index_csv.parent.resolve(),
    }
    unsafe = resolved in protected
    unsafe = unsafe or _is_relative_to(metrics_csv.resolve(), resolved)
    unsafe = unsafe or _is_relative_to(vis_index_csv.resolve(), resolved)
    if unsafe:
        raise ValueError(
            f"Refusing to delete protected output_dir={output_dir}. "
            "Choose a dedicated bad-sample directory."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter NavSIM evaluation visualizations by PDM score and collect bad samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics_csv",
        required=True,
        type=Path,
        help="Path to navsim_test_metrics.csv from scripts/evaluate_navsim.py.",
    )
    parser.add_argument(
        "--vis_index_csv",
        type=Path,
        default=None,
        help="Path to vis/index.csv. Defaults to <metrics_csv parent>/vis/index.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for collected bad samples. Defaults to <metrics_csv parent>/bad_samples.",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="Threshold used to decide bad samples.",
    )
    parser.add_argument(
        "--metric",
        default=TEST_SCORE_METRIC,
        help="Metric name to filter. NavSIM test PDM score is 'score'. 'pdm_score' is accepted as an alias when score exists.",
    )
    parser.add_argument(
        "--comparison",
        choices=("lt", "le", "gt", "ge"),
        default="lt",
        help="Bad-sample rule: lt means metric < threshold; le means <=; gt means >; ge means >=.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on copied bad samples after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag; output_dir is refreshed on every non-dry run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be selected without copying files.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    metrics_csv = args.metrics_csv.expanduser()
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics_csv not found: {metrics_csv}")

    vis_index_csv = args.vis_index_csv.expanduser() if args.vis_index_csv else metrics_csv.parent / "vis" / "index.csv"
    if not vis_index_csv.is_file():
        raise FileNotFoundError(f"vis_index_csv not found: {vis_index_csv}")

    output_dir = args.output_dir.expanduser() if args.output_dir else metrics_csv.parent / "bad_samples"

    metric_rows = _load_metrics(metrics_csv)
    filter_metric = _resolve_filter_metric(args.metric, metric_rows)
    if not args.dry_run:
        _refresh_output_dir(output_dir, metrics_csv=metrics_csv, vis_index_csv=vis_index_csv)

    vis_rows = _load_visualization_rows(vis_index_csv)
    bases = [
        Path.cwd(),
        metrics_csv.parent,
        metrics_csv.parent.parent,
        vis_index_csv.parent,
        vis_index_csv.parent.parent,
    ]

    selected: list[dict[str, object]] = []
    seen_dirs: set[Path] = set()
    for vis_row in vis_rows:
        sample_id = _row_id(vis_row)
        if not sample_id or sample_id not in metric_rows:
            continue
        metric_row = metric_rows[sample_id]
        metrics = metric_row["metrics"]
        assert isinstance(metrics, dict)
        value = _metric_value(metrics, filter_metric)
        if value is None:
            continue
        if not _is_bad_sample(float(value), float(args.threshold), args.comparison):
            continue

        token = str(metric_row.get("token") or sample_id)
        folder_name = f"{_safe_name(token)}_{_safe_name(filter_metric, max_len=40)}_{float(value):.4f}"
        sample_dir = output_dir / folder_name
        if sample_dir in seen_dirs:
            sample_dir = output_dir / f"{folder_name}_idx_{_safe_name(metric_row.get('idx', ''), max_len=30)}"
        seen_dirs.add(sample_dir)

        if args.overwrite and sample_dir.exists() and not args.dry_run:
            shutil.rmtree(sample_dir)

        copied = _copy_visualizations(
            vis_row=vis_row,
            sample_dir=sample_dir,
            bases=bases,
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_metrics(sample_dir, metric_row, filter_metric, float(value))

        selected.append(
            {
                "idx": metric_row.get("idx", ""),
                "token": token,
                "log_name": metric_row.get("log_name", ""),
                "metric": filter_metric,
                "value": f"{float(value):.10g}",
                "sample_dir": str(sample_dir),
                "bev_path": copied.get("bev_path", ""),
                "trajectory_path": copied.get("trajectory_path", ""),
                "world_model_path": copied.get("world_model_path", ""),
            }
        )
        if args.max_samples is not None and len(selected) >= int(args.max_samples):
            break

    manifest_path = None
    if not args.dry_run:
        manifest_path = _write_manifest(selected, output_dir)

    print(f"metrics_csv: {metrics_csv}")
    print(f"vis_index_csv: {vis_index_csv}")
    print(f"metric: {filter_metric}")
    if filter_metric != args.metric:
        print(f"requested_metric: {args.metric}")
    print(f"rule: {args.comparison} {args.threshold}")
    print(f"selected_bad_samples: {len(selected)}")
    if manifest_path is not None:
        print(f"output_dir: {output_dir}")
        print(f"manifest: {manifest_path}")
    if args.dry_run:
        for row in selected[:20]:
            idx = row.get("idx", "")
            token = row.get("token", "")
            metric = row.get("metric", "")
            value = row.get("value", "")
            print(f"{idx} {token} {metric}={value}")


if __name__ == "__main__":
    main()

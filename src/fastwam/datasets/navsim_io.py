from __future__ import annotations

import csv
import io
import lzma
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


def is_obs_path(path: Optional[str | Path]) -> bool:
    return str(path).strip().lower().startswith("obs://") if path is not None else False


def normalize_storage_mode(storage_mode: str, *paths: Optional[str | Path]) -> str:
    key = str(storage_mode).strip().lower()
    if key not in {"local", "obs"}:
        raise ValueError(f"Unsupported NavSIM storage mode: {storage_mode!r}. Expected 'local' or 'obs'.")
    if key == "local" and any(is_obs_path(path) for path in paths):
        return "obs"
    return key


class MoxingNavsimIO:
    """Small moxing wrapper used only when NavSIM data lives on OBS."""

    def __init__(self):
        try:
            import moxing as mox  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "NavSIM storage_mode=obs requires the `moxing` package in the runtime environment. "
                "Install/use a ModelArts environment with moxing and OBS credentials configured."
            ) from exc
        self.mox = mox
        self.file = mox.file

    @staticmethod
    def join(root: str | Path, *parts: Any) -> str:
        value = str(root).rstrip("/")
        for part in parts:
            piece = str(part).strip("/")
            if piece:
                value = f"{value}/{piece}"
        return value

    @staticmethod
    def basename(path: str | Path) -> str:
        return str(path).rstrip("/").split("/")[-1]

    def list_directory(self, path: str | Path) -> list[str]:
        root = str(path).rstrip("/")
        entries = self.file.list_directory(root)
        normalized: list[str] = []
        for entry in entries:
            value = str(entry).rstrip("/")
            if not value or value in {".", ".."}:
                continue
            if value.startswith("obs://") or value.startswith("/"):
                normalized.append(value)
            else:
                normalized.append(self.join(root, value))
        return normalized

    def exists(self, path: str | Path) -> bool:
        exists = getattr(self.file, "exists", None)
        if exists is None:
            return True
        return bool(exists(str(path)))

    def read_bytes(self, path: str | Path) -> bytes:
        value = str(path)
        read = getattr(self.file, "read", None)
        if read is not None:
            try:
                payload = read(value, binary=True)
                if isinstance(payload, bytes):
                    return payload
                if isinstance(payload, bytearray):
                    return bytes(payload)
                if isinstance(payload, str):
                    return payload.encode("utf-8")
            except TypeError:
                pass
            except Exception:
                pass

        handle = self.file.File(value, "rb")
        try:
            payload = handle.read()
        finally:
            close = getattr(handle, "close", None)
            if close is not None:
                close()
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, str):
            return payload.encode("utf-8")
        raise TypeError(f"Unsupported moxing read payload type for {value}: {type(payload)}")

    def read_text(self, path: str | Path, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding)

    def read_pickle(self, path: str | Path) -> Any:
        return pickle.loads(self.read_bytes(path))

    def read_image_array(self, path: str | Path) -> np.ndarray:
        with Image.open(io.BytesIO(self.read_bytes(path))) as image:
            return np.array(image)


class MoxingMetricCacheLoader:
    """MetricCacheLoader-compatible reader for metric cache files stored on OBS."""

    def __init__(self, cache_path: str | Path, file_name: str = "metric_cache.pkl", io_client: MoxingNavsimIO | None = None):
        self.cache_path = str(cache_path).rstrip("/")
        self.file_name = str(file_name)
        self.io = io_client or MoxingNavsimIO()
        self.metric_cache_paths = self._load_metric_cache_paths()

    def _normalize_metadata_cache_path(self, metadata_path: str) -> str:
        value = metadata_path.strip()
        if is_obs_path(value):
            return value

        root_name = self.cache_path.rstrip("/").split("/")[-1]
        parts = value.strip("/").split("/")
        if root_name in parts:
            idx = len(parts) - 1 - parts[::-1].index(root_name)
            rel_path = "/".join(parts[idx + 1 :])
            return self.io.join(self.cache_path, rel_path)
        return self.io.join(self.cache_path, value.lstrip("/"))

    def _load_metric_cache_paths(self) -> dict[str, str]:
        metadata_dir = self.io.join(self.cache_path, "metadata")
        metadata_files = [path for path in self.io.list_directory(metadata_dir) if self.io.basename(path).endswith(".csv")]
        if not metadata_files:
            raise FileNotFoundError(f"No NAVSIM metric cache metadata csv found in {metadata_dir}")

        text = self.io.read_text(sorted(metadata_files)[0])
        rows = list(csv.reader(text.splitlines()))
        cache_paths: dict[str, str] = {}
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            cache_path = self._normalize_metadata_cache_path(row[0])
            token = cache_path.rstrip("/").split("/")[-2]
            cache_paths[token] = cache_path
        return cache_paths

    @property
    def tokens(self) -> list[str]:
        return list(self.metric_cache_paths.keys())

    def __len__(self) -> int:
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int):
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str):
        path = self.metric_cache_paths[token]
        with lzma.LZMAFile(io.BytesIO(self.io.read_bytes(path)), "rb") as fp:
            return pickle.load(fp)


def build_metric_cache_loader(metric_cache_path: str | Path, storage_mode: str = "local", io_client: MoxingNavsimIO | None = None):
    mode = normalize_storage_mode(storage_mode, metric_cache_path)
    if mode == "obs":
        return MoxingMetricCacheLoader(metric_cache_path, io_client=io_client)

    from navsim.common.dataloader import MetricCacheLoader

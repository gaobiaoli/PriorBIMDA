from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import yaml


class Config(dict):
    """Small recursively attribute-accessible configuration object."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = _convert(raw)
    cfg.config_path = str(path)
    cfg.project_root = str(path.parent.parent)
    return cfg


def resolve_project_path(cfg: Config, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(cfg.project_root) / path
    return path.resolve()


def resolve_slabim_root(cfg: Config) -> Path:
    """Resolve SLABIM with an environment override for relocated workspaces."""
    value = os.environ.get("BIM_PRIORDA3_SLABIM_ROOT", str(cfg.data.slabim_root))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(cfg.project_root) / path
    return path.resolve()

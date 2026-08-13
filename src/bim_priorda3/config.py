from __future__ import annotations

import os
from pathlib import Path
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


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance involving {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Config must contain a mapping: {path}")
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return raw
    parent = Path(parent_value).expanduser()
    if not parent.is_absolute():
        parent = path.parent / parent
    inherited = _load_raw_config(parent, seen | {path})
    return _merge_dicts(inherited, raw)


def _find_project_root(config_path: Path) -> Path:
    """Find the package root so generated configs may live in nested folders."""
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent.parent


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    raw = _load_raw_config(path, set())
    cfg = _convert(raw)
    cfg.config_path = str(path)
    cfg.project_root = str(_find_project_root(path))
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

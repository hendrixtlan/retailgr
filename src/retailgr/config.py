"""Configuration loading for the RetailGR Stage 1 pipeline."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_GRANULARITY_CONFIG = REPO_ROOT / "configs" / "granularity.yaml"


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return ``base``."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _resolve_path(value: str, base: Path) -> str:
    """Make a config path absolute, relative to the repository root."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


@dataclass
class Config:
    """Pipeline configuration plus the granularity rules."""

    pipeline: dict[str, Any] = field(default_factory=dict)
    granularity: dict[str, Any] = field(default_factory=dict)
    root: Path = REPO_ROOT

    @classmethod
    def load(
        cls,
        pipeline_path: str | os.PathLike[str] | None = None,
        granularity_path: str | os.PathLike[str] | None = None,
        overrides: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> Config:
        root = root or REPO_ROOT
        pipeline = _read_yaml(pipeline_path or DEFAULT_PIPELINE_CONFIG)
        granularity = _read_yaml(granularity_path or DEFAULT_GRANULARITY_CONFIG)
        if overrides:
            _deep_update(pipeline, copy.deepcopy(overrides))

        # Paths in the YAML are relative to the repository root, not to the
        # shell's working directory.
        pipeline["warehouse"]["root"] = _resolve_path(pipeline["warehouse"]["root"], root)
        pipeline["dataset"]["raw_path"] = _resolve_path(pipeline["dataset"]["raw_path"], root)
        if pipeline.get("spark", {}).get("local_dir"):
            pipeline["spark"]["local_dir"] = _resolve_path(pipeline["spark"]["local_dir"], root)

        return cls(pipeline=pipeline, granularity=granularity, root=root)

    # -- convenience accessors -------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a nested value, e.g. ``cfg.get("split.train_frac")``."""
        node: Any = self.pipeline
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def warehouse_backend(self) -> str:
        return str(self.get("warehouse.backend", "parquet")).lower()

    @property
    def warehouse_root(self) -> Path:
        return Path(self.get("warehouse.root"))

    @property
    def raw_path(self) -> Path:
        return Path(self.get("dataset.raw_path"))

    @property
    def dataset_name(self) -> str:
        return str(self.get("dataset.name", "synthetic")).lower()

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"


def load_model_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a model preset from ``configs/model``."""
    candidate = Path(path)
    if not candidate.exists():
        candidate = REPO_ROOT / "configs" / "model" / Path(path).name
    if not candidate.exists():
        raise FileNotFoundError(f"model config not found: {path}")
    return _read_yaml(candidate)

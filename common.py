from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym


ENV_ID = "ALE/Pong-v5"


@dataclass(frozen=True)
class TrainPaths:
    root: Path
    models: Path
    logs: Path
    videos: Path
    live: Path


def ensure_ale_installed() -> Any:
    try:
        import ale_py  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing ale-py. Install dependencies with:\n"
            "pip install -r requirements.txt\n"
            "If ROMs are still missing, run:\n"
            "AutoROM --accept-license"
        ) from exc

    gym.register_envs(ale_py)
    return ale_py


def prepare_paths(root: Path) -> TrainPaths:
    models = root / "artifacts" / "models"
    logs = root / "artifacts" / "logs"
    videos = root / "artifacts" / "videos"
    live = root / "artifacts" / "live"
    for path in (models, logs, videos, live):
        path.mkdir(parents=True, exist_ok=True)
    return TrainPaths(root=root, models=models, logs=logs, videos=videos, live=live)

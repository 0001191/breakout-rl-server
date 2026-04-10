from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Breakout training in the background on a server.")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--preview-freq", type=int, default=20_000)
    parser.add_argument("--status-freq", type=int, default=2_000)
    parser.add_argument("--stream-freq", type=int, default=32)
    parser.add_argument("--display-fps", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    live_dir = ROOT / "artifacts" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    runner_log = live_dir / "launcher.log"

    cmd = [
        sys.executable,
        "train_breakout.py",
        "--root-dir",
        ".",
        "--total-timesteps",
        str(args.total_timesteps),
        "--device",
        args.device,
        "--n-envs",
        str(args.n_envs),
        "--buffer-size",
        str(args.buffer_size),
        "--preview-freq",
        str(args.preview_freq),
        "--status-freq",
        str(args.status_freq),
        "--stream-freq",
        str(args.stream_freq),
        "--display-fps",
        str(args.display_fps),
    ]

    with runner_log.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    print(f"Training started in background. PID={process.pid}")
    print(f"Launcher log: {runner_log}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as iio2
import imageio.v3 as iio
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from common import ENV_ID, ensure_ale_installed, prepare_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Pong DQN agent with Gymnasium + SB3.")
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--status-freq", type=int, default=2_000)
    parser.add_argument("--preview-freq", type=int, default=20_000)
    parser.add_argument("--preview-steps", type=int, default=300)
    parser.add_argument("--stream-freq", type=int, default=32)
    parser.add_argument("--display-fps", type=float, default=12.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root-dir", default=".")
    return parser.parse_args()


def make_train_env(env_id: str, n_envs: int, seed: int, frame_stack: int, render_training: bool):
    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"render_mode": "rgb_array"} if render_training else None,
        wrapper_kwargs={"terminal_on_life_loss": True},
    )
    env = VecFrameStack(env, n_stack=frame_stack)
    return VecTransposeImage(env)


def make_eval_env(env_id: str, seed: int, frame_stack: int):
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        wrapper_kwargs={"terminal_on_life_loss": False},
    )
    env = VecFrameStack(env, n_stack=frame_stack)
    return VecTransposeImage(env)


def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_image(path: Path, frame) -> None:
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    iio.imwrite(tmp, frame)
    tmp.replace(path)


def configure_logging(root: Path) -> Path:
    live_dir = root / "artifacts" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    log_path = live_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


class LiveDashboardCallback(BaseCallback):
    def __init__(
        self,
        *,
        live_dir: Path,
        env_id: str,
        seed: int,
        frame_stack: int,
        status_freq: int,
        preview_freq: int,
        preview_steps: int,
        stream_freq: int,
        display_fps: float,
        total_timesteps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.live_dir = live_dir
        self.env_id = env_id
        self.seed = seed
        self.frame_stack = frame_stack
        self.status_freq = max(status_freq, 1)
        self.preview_freq = max(preview_freq, 1)
        self.preview_steps = max(preview_steps, 60)
        self.stream_freq = max(stream_freq, 1)
        self.display_fps = max(display_fps, 0.0)
        self.total_timesteps = total_timesteps
        self.started_at = 0.0
        self.last_stream_at = 0.0
        self.preview_env = None
        self.preview_obs = None
        self.last_stream_timestep = -1
        self.history: list[dict[str, float | int | None]] = []
        self.status_path = self.live_dir / "status.json"
        self.history_path = self.live_dir / "history.json"
        self.preview_path = self.live_dir / "preview.png"
        self.preview_gif_path = self.live_dir / "preview.gif"
        self.stream_frame_path = self.live_dir / "preview_live.jpg"

    def _init_callback(self) -> None:
        self.started_at = time.time()
        self.last_stream_at = self.started_at
        self.preview_env = make_atari_env(
            self.env_id,
            n_envs=1,
            seed=self.seed + 20_000,
            env_kwargs={"render_mode": "rgb_array"},
            wrapper_kwargs={"terminal_on_life_loss": False},
        )
        self.preview_env = VecFrameStack(self.preview_env, n_stack=self.frame_stack)
        self.preview_env = VecTransposeImage(self.preview_env)
        obs = self.preview_env.reset()
        self.preview_obs = obs[0] if isinstance(obs, tuple) else obs
        self._write_status(phase="starting")
        self._advance_live_stream(force=True)
        self._refresh_preview()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.status_freq == 0:
            self._write_status(phase="training")
        self._advance_live_stream()
        if self.num_timesteps % self.preview_freq == 0:
            self._refresh_preview()
        return True

    def _on_training_end(self) -> None:
        self._write_status(phase="finished")
        self._advance_live_stream(force=True)
        self._refresh_preview()
        if self.preview_env is not None:
            self.preview_env.close()

    def _episode_stats(self) -> tuple[list[dict[str, float]], float | None, float | None]:
        raw = list(self.model.ep_info_buffer) if getattr(self.model, "ep_info_buffer", None) else []
        rewards = [float(item["r"]) for item in raw if "r" in item]
        lengths = [float(item["l"]) for item in raw if "l" in item]
        mean_reward = sum(rewards) / len(rewards) if rewards else None
        mean_length = sum(lengths) / len(lengths) if lengths else None
        recent = [{"reward": float(item.get("r", 0.0)), "length": float(item.get("l", 0.0))} for item in raw[-12:]]
        return recent, mean_reward, mean_length

    def _write_status(self, *, phase: str) -> None:
        recent, mean_reward, mean_length = self._episode_stats()
        elapsed = max(time.time() - self.started_at, 1e-6)
        fps = round(self.num_timesteps / elapsed, 2)
        exploration_rate = float(getattr(self.model, "exploration_rate", 0.0))
        loss = None
        if hasattr(self.model, "logger") and "train/loss" in self.model.logger.name_to_value:
            loss = float(self.model.logger.name_to_value["train/loss"])

        point = {
            "timesteps": int(self.num_timesteps),
            "progress": round(self.num_timesteps / max(self.total_timesteps, 1), 6),
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "exploration_rate": exploration_rate,
            "loss": loss,
            "fps": fps,
        }
        if not self.history or self.history[-1]["timesteps"] != point["timesteps"]:
            self.history.append(point)
            self.history = self.history[-240:]

        payload = {
            "phase": phase,
            "timesteps": int(self.num_timesteps),
            "total_timesteps": int(self.total_timesteps),
            "progress_percent": round(100 * self.num_timesteps / max(self.total_timesteps, 1), 2),
            "elapsed_seconds": round(elapsed, 1),
            "fps": fps,
            "exploration_rate": exploration_rate,
            "loss": loss,
            "mean_reward_100": mean_reward,
            "mean_length_100": mean_length,
            "recent_episodes": recent,
            "last_preview_path": self.preview_gif_path.name if self.preview_gif_path.exists() else (self.preview_path.name if self.preview_path.exists() else None),
            "stream_path": self.stream_frame_path.name if self.stream_frame_path.exists() else None,
        }
        atomic_write_text(self.status_path, json.dumps(payload, indent=2, ensure_ascii=False))
        atomic_write_text(self.history_path, json.dumps(self.history, indent=2, ensure_ascii=False))

    def _advance_live_stream(self, *, force: bool = False) -> None:
        if not force and self.last_stream_timestep >= 0 and self.num_timesteps - self.last_stream_timestep < self.stream_freq:
            return

        images = []
        if self.training_env is not None:
            try:
                images = self.training_env.get_images()
            except Exception:
                images = []

        if not images and self.preview_env is not None and self.preview_obs is not None:
            action, _ = self.model.predict(self.preview_obs, deterministic=True)
            obs, rewards, dones, infos = self.preview_env.step(action)
            self.preview_obs = obs
            images = self.preview_env.get_images()
            if bool(dones[0]):
                reset_obs = self.preview_env.reset()
                self.preview_obs = reset_obs[0] if isinstance(reset_obs, tuple) else reset_obs

        if images:
            atomic_write_image(self.stream_frame_path, images[0])

        if self.display_fps > 0 and getattr(self.training_env, "num_envs", 1) == 1:
            now = time.time()
            target = self.last_stream_at + (1.0 / self.display_fps)
            delay = target - now
            if delay > 0:
                time.sleep(delay)
                now = time.time()
            self.last_stream_at = now

        self.last_stream_timestep = self.num_timesteps

    def _refresh_preview(self) -> None:
        if self.preview_env is None:
            return
        obs = self.preview_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        frame = None
        frames: list = []
        for _ in range(self.preview_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.preview_env.step(action)
            images = self.preview_env.get_images()
            if images:
                frame = images[0]
                frames.append(frame)
            if bool(dones[0]):
                break
        if frame is not None:
            atomic_write_image(self.preview_path, frame)
        if frames:
            stride = max(len(frames) // 80, 1)
            sampled = frames[::stride]
            iio2.mimsave(self.preview_gif_path, sampled, format="GIF", duration=0.05, loop=0)
        reset_obs = self.preview_env.reset()
        self.preview_obs = reset_obs[0] if isinstance(reset_obs, tuple) else reset_obs
        self._advance_live_stream(force=True)


def main() -> None:
    ensure_ale_installed()
    args = parse_args()

    root = Path(args.root_dir).resolve()
    log_path = configure_logging(root)
    paths = prepare_paths(root)
    env_slug = args.env_id.split("/")[-1].split("-")[0].lower()
    run_name = f"{env_slug}_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_model_dir = paths.models / run_name
    run_model_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Starting training run %s", run_name)
    logging.info("Live log file: %s", log_path)

    train_env = make_train_env(args.env_id, args.n_envs, args.seed, args.frame_stack, render_training=args.display_fps > 0)
    eval_env = make_eval_env(args.env_id, args.seed + 10_000, args.frame_stack)

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        optimize_memory_usage=False,
        tensorboard_log=str(paths.logs),
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(run_model_dir / "checkpoints"),
        name_prefix="breakout_dqn",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir / "best_model"),
        log_path=str(run_model_dir / "eval_metrics"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    live_callback = LiveDashboardCallback(
        live_dir=paths.live,
        env_id=args.env_id,
        seed=args.seed,
        frame_stack=args.frame_stack,
        status_freq=args.status_freq,
        preview_freq=args.preview_freq,
        preview_steps=args.preview_steps,
        stream_freq=args.stream_freq,
        display_fps=args.display_fps,
        total_timesteps=args.total_timesteps,
    )

    config = {
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "frame_stack": args.frame_stack,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "gamma": args.gamma,
        "status_freq": args.status_freq,
        "preview_freq": args.preview_freq,
        "preview_steps": args.preview_steps,
        "stream_freq": args.stream_freq,
        "display_fps": args.display_fps,
        "device": args.device,
    }
    (run_model_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback, live_callback],
            progress_bar=True,
            tb_log_name=run_name,
        )
        model.save(run_model_dir / "final_model")
        logging.info("Training complete. Final model saved to %s", run_model_dir / "final_model.zip")
        print(f"Training complete. Saved final model to: {run_model_dir / 'final_model.zip'}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

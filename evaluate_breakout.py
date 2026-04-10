from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

from common import ENV_ID, ensure_ale_installed, prepare_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or record a trained Breakout DQN.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-length", type=int, default=6_000)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_eval_env(env_id: str, seed: int, frame_stack: int, record_video: bool, root_dir: Path, video_length: int):
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": "rgb_array"} if record_video else None,
        wrapper_kwargs={"terminal_on_life_loss": False},
    )
    env = VecFrameStack(env, n_stack=frame_stack)
    env = VecTransposeImage(env)
    if record_video:
        paths = prepare_paths(root_dir)
        env = VecVideoRecorder(
            env,
            video_folder=str(paths.videos),
            record_video_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix="breakout_eval",
        )
    return env


def main() -> None:
    ensure_ale_installed()
    args = parse_args()

    root = Path(args.root_dir).resolve()
    model = DQN.load(args.model_path, device=args.device)
    env = make_eval_env(args.env_id, args.seed, args.frame_stack, args.record_video, root, args.video_length)

    try:
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.episodes,
            deterministic=args.deterministic,
            return_episode_rewards=False,
        )
        print(f"Mean reward over {args.episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_rewards = []
        current_reward = 0.0
        for _ in range(args.video_length):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(action)
            current_reward += float(np.asarray(rewards).reshape(-1)[0])
            if bool(np.asarray(dones).reshape(-1)[0]):
                episode_rewards.append(current_reward)
                current_reward = 0.0
        if current_reward:
            episode_rewards.append(current_reward)
        if episode_rewards:
            print("Recorded rollout rewards:", ", ".join(f"{reward:.1f}" for reward in episode_rewards))
        if args.record_video:
            print(f"Saved video to: {(prepare_paths(root).videos).resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

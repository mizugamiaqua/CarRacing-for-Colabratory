"""学習済みモデルを評価し、周回(全タイル通過)の可否と動画を出力する。

使い方: python -m carracing.evaluate --model <zip> --episodes 10 --video out.mp4
"""
import argparse

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from .config import ENV_ID, FRAME_STACK


class LapInfo(gym.Wrapper):
    """終了時に全タイル通過(=1周完走)かを info["lap_complete"] に入れる。"""

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if term or trunc:
            u = self.env.unwrapped
            info["lap_complete"] = u.tile_visited_count == len(u.track)
        return obs, r, term, trunc, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--video", default=None, help="最初の1エピソードを mp4 保存")
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()

    raw = gym.make(ENV_ID, render_mode="rgb_array")
    venv = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: LapInfo(raw)]), FRAME_STACK))
    model = PPO.load(a.model, device="cpu")

    scores, laps = [], 0
    for ep in range(a.episodes):
        venv.seed(a.seed + ep)
        obs = venv.reset()
        frames, total, done = [], 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, d, infos = venv.step(action)
            total += float(r[0]); done = bool(d[0])
            if a.video and ep == 0:
                frames.append(raw.render())
        lap = bool(infos[0].get("lap_complete", False))
        laps += lap
        scores.append(total)
        print(f"episode {ep}: score={total:.1f} lap_complete={lap}")
        if a.video and ep == 0:
            imageio.mimsave(a.video, frames, fps=50)
    print(f"mean={np.mean(scores):.1f} std={np.std(scores):.1f} "
          f"min={np.min(scores):.1f} lap_rate={laps}/{a.episodes}")


if __name__ == "__main__":
    main()

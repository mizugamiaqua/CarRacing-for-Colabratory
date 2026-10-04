"""PPO で CarRacing を学習する。中断しても --out の最新チェックポイントから再開できる。

使い方: python -m carracing.train --out /content/drive/MyDrive/carracing --steps 4000000
"""
import argparse, glob, os, re, urllib.request

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from . import config as C
from .envs import make_env, wrap_normalize


def linear(start):
    return lambda progress_remaining: progress_remaining * start


def latest_checkpoint(out):
    best, best_n = None, -1
    for p in glob.glob(os.path.join(out, "ckpt", "ppo_*_steps.zip")):
        n = int(re.search(r"ppo_(\d+)_steps", p).group(1))
        if n > best_n:
            best, best_n = p, n
    return best, best_n


def fetch(path_or_url, dst_dir):
    """URL ならダウンロードし、ローカルパスならそのまま返す。"""
    if not path_or_url.startswith(("http://", "https://")):
        return path_or_url
    dst = os.path.join(dst_dir, "init_model.zip")
    if not os.path.exists(dst):
        print("download", path_or_url)
        urllib.request.urlretrieve(path_or_url, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/carracing")
    ap.add_argument("--steps", type=int, default=C.TOTAL_TIMESTEPS)
    ap.add_argument("--n-envs", type=int, default=C.N_ENVS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="既存の学習済み model.zip (パス or URL) の重みから学習を開始 (転移/追加学習)。"
                         "--out に既にチェックポイントがあればそちらの再開を優先")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="--out に final_model.zip があれば学習せず終了")
    ap.add_argument("--ckpt-every", type=int, default=200_000)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.skip_if_done and os.path.exists(os.path.join(a.out, "final_model.zip")):
        print("final_model.zip が既にあるため学習をスキップします")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ckpt, done = latest_checkpoint(a.out)
    stats = os.path.join(a.out, "ckpt", f"ppo_vecnormalize_{done}_steps.pkl") if ckpt else None
    venv = wrap_normalize(make_env(a.n_envs, a.seed),
                          stats if stats and os.path.exists(stats) else None)

    if ckpt:
        print(f"resume from {ckpt}")
        model = PPO.load(ckpt, env=venv, device=device,
                         custom_objects={"learning_rate": linear(C.LR_START)})
    elif a.init_from:
        src = fetch(a.init_from, a.out)
        print(f"warm start from {src}")
        model = PPO.load(src, env=venv, device=device, tensorboard_log=os.path.join(a.out, "tb"),
                         custom_objects={"learning_rate": linear(C.LR_START)})
        model.num_timesteps = 0  # この実行分のステップ数として数え直す
    else:
        model = PPO("CnnPolicy", venv, learning_rate=linear(C.LR_START), seed=a.seed,
                    device=device, verbose=1,
                    tensorboard_log=os.path.join(a.out, "tb"), **C.PPO_KWARGS)

    eval_env = wrap_normalize(make_env(1, a.seed + 100, subproc=False), training=False)
    # eval 側にも学習中の報酬統計は不要だが、VecNormalize は同期不要 (報酬のみ)
    cbs = [
        CheckpointCallback(max(a.ckpt_every // a.n_envs, 1), os.path.join(a.out, "ckpt"),
                           name_prefix="ppo", save_vecnormalize=True),
        EvalCallback(eval_env, n_eval_episodes=3, eval_freq=max(250_000 // a.n_envs, 1),
                     best_model_save_path=os.path.join(a.out, "best"),
                     log_path=os.path.join(a.out, "eval"), deterministic=True),
    ]
    remaining = a.steps - model.num_timesteps
    if remaining > 0:
        model.learn(remaining, callback=cbs, reset_num_timesteps=False,
                    tb_log_name="ppo", progress_bar=True)
    model.save(os.path.join(a.out, "final_model"))
    venv.save(os.path.join(a.out, "vecnormalize.pkl"))
    print("saved to", a.out)


if __name__ == "__main__":
    main()

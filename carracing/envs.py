import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage,
)

from .config import ENV_ID, FRAME_STACK


def make_env(n_envs, seed=0, subproc=True, **env_kwargs):
    cls = SubprocVecEnv if (subproc and n_envs > 1) else DummyVecEnv
    venv = make_vec_env(ENV_ID, n_envs=n_envs, seed=seed, vec_env_cls=cls,
                        env_kwargs=env_kwargs)
    venv = VecFrameStack(venv, FRAME_STACK)
    return VecTransposeImage(venv)


def wrap_normalize(venv, stats_path=None, training=True):
    """報酬のみ正規化 (観測は CnnPolicy 側で /255 される)。"""
    if stats_path:
        venv = VecNormalize.load(stats_path, venv)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    venv.training = training
    venv.norm_reward = training
    return venv

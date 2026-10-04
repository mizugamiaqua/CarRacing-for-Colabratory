"""ハイパーパラメータ設定 (rl-baselines3-zoo の CarRacing PPO 設定をベース)。"""

ENV_ID = "CarRacing-v3"
N_ENVS = 8
FRAME_STACK = 2
TOTAL_TIMESTEPS = 4_000_000  # 1周完走が安定して出る目安。短縮版は 2_000_000

PPO_KWARGS = dict(
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    use_sde=True,
    sde_sample_freq=4,
    policy_kwargs=dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=__import__("torch").nn.ReLU,
        net_arch=dict(pi=[256], vf=[256]),
    ),
)
LR_START = 1e-4  # 線形に 0 まで減衰

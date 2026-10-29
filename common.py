"""Shared helpers: config loading, env construction, algorithm construction."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", Path(path).stem)
    return cfg


def run_dir(cfg, seed):
    return RUNS / f"{cfg['name']}_s{seed}"


def make_env(cfg, render_mode=None, **render_kwargs):
    import gymnasium as gym
    from sb3_contrib.common.wrappers import TimeFeatureWrapper

    if cfg["env_id"].startswith("Panda"):
        import panda_gym  # noqa: F401  (registers the envs)
    else:
        os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless rendering
        import envs  # noqa: F401
    if render_mode is not None:  # panda-gym 3 rejects render_mode=None (its default rgb_array is headless)
        render_kwargs["render_mode"] = render_mode
    env = gym.make(cfg["env_id"], **cfg.get("env_kwargs", {}), **render_kwargs)
    return TimeFeatureWrapper(env)


def make_model(cfg, env, seed):
    import numpy as np
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.noise import NormalActionNoise

    kw = dict(cfg.get("algo_kwargs", {}))
    if "action_noise_sigma" in cfg:
        n = env.action_space.shape[0]
        kw["action_noise"] = NormalActionNoise(np.zeros(n), cfg["action_noise_sigma"] * np.ones(n))
    return algo_class(cfg)("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=cfg["her"], policy_kwargs=cfg.get("policy_kwargs"), seed=seed,
                verbose=0, **kw)


def algo_class(cfg):
    from stable_baselines3 import DDPG, SAC, TD3
    return {"ddpg": DDPG, "td3": TD3, "sac": SAC}[cfg["algo"]]

"""Video of the trained policy: N deterministic episodes, headless, MP4 + GIF.

python record.py --config configs/ur5e_ddpg.yaml --seed 0 [--episodes 10]
Writes results/videos/<name>_s<seed>.mp4 and .gif.
"""
import argparse

import imageio.v2 as imageio
import numpy as np
import torch

from common import ROOT, algo_class, load_config, make_env, run_dir
from evaluate import TEST_SEED0

HOLD = 6  # extra frames shown at the end of each episode
PANDA_CAMERA = dict(render_width=480, render_height=360, render_target_position=[0.0, 0.0, 0.1],
                    render_distance=1.1, render_yaw=50, render_pitch=-25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model", default="final_model.zip")
    a = p.parse_args()
    torch.set_num_threads(1)

    cfg = load_config(a.config)
    run = run_dir(cfg, a.seed)
    env = make_env(cfg, render_mode="rgb_array", **(PANDA_CAMERA if cfg["env_id"].startswith("Panda") else {}))
    model = algo_class(cfg).load(run / a.model, env=env)
    frames, results = [], []
    for i in range(a.episodes):
        obs, _ = env.reset(seed=TEST_SEED0 + i)
        frames.append(env.render())
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frames.append(env.render())
        frames += [frames[-1]] * HOLD
        results.append(info["is_success"])
        print(f"episode {i}: success={info['is_success']}", flush=True)
    out = ROOT / "results" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    fps = env.metadata.get("render_fps") or 25
    imageio.mimsave(out / f"{run.name}.mp4", frames, fps=fps, macro_block_size=8)
    small = [f[::2, ::2] for f in frames]  # half resolution keeps the GIF small enough for the README
    imageio.mimsave(out / f"{run.name}.gif", small, duration=1000 / fps, loop=0)
    print(f"{np.mean(results):.0%} success; wrote {out / run.name}.mp4/.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()

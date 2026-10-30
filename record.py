"""Video of the trained policy: N deterministic episodes, headless, MP4 + GIF.

python record.py --config configs/ur5e_ddpg.yaml --seed 0 [--episodes 10] [--scripted | --side-by-side]
Writes results/videos/<name>_s<seed>.mp4 and .gif; --scripted records the no-learning rule (<name>_scripted),
--side-by-side puts the scripted rule (left) next to the RL policy (right) on the same episodes (<name>_s<seed>_vs).
"""
import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, algo_class, load_config, make_env, run_dir
from evaluate import TEST_SEED0

HOLD = 8  # extra frames shown at the end of each episode
SLOWDOWN = 3  # reaches take only 2-5 control steps, so play at 1/3 of real time
PANDA_CAMERA = dict(render_width=480, render_height=360, render_target_position=[0.0, 0.0, 0.1],
                    render_distance=1.1, render_yaw=50, render_pitch=-25)


def episode_frames(env, act, seed):
    """Frames of one episode (plus HOLD copies of the last) and its outcome."""
    obs, _ = env.reset(seed=seed)
    frames, done = [env.render()], False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(obs))
        done = terminated or truncated
        frames.append(env.render())
    return frames + [frames[-1]] * HOLD, info


def label(frame, text, color):
    img = Image.fromarray(frame)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((10, 8), text, fill=color, font=font)
    return np.asarray(img)


def outcome(info):
    if info.get("collision"):
        return "COLLISION", (200, 30, 30)
    return ("reached", (20, 140, 40)) if info["is_success"] else ("missed", (200, 120, 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model", default="final_model.zip")
    p.add_argument("--scripted", action="store_true", help="record the scripted no-learning rule instead")
    p.add_argument("--side-by-side", action="store_true", help="scripted (left) vs RL policy (right)")
    a = p.parse_args()
    torch.set_num_threads(1)

    cfg = load_config(a.config)
    run = run_dir(cfg, a.seed)
    env = make_env(cfg, render_mode="rgb_array", **(PANDA_CAMERA if cfg["env_id"].startswith("Panda") else {}))
    from baseline import scripted_action
    scripted = lambda obs: scripted_action(env.unwrapped, obs)
    if not a.scripted:
        model = algo_class(cfg).load(run / a.model, env=env)
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    frames, results = [], []
    for i in range(a.episodes):
        if a.side_by_side:
            left, li = episode_frames(env, scripted, TEST_SEED0 + i)
            right, ri = episode_frames(env, policy, TEST_SEED0 + i)
            ends = len(left) - HOLD - 1, len(right) - HOLD - 1  # frame index where each episode ended
            n = max(len(left), len(right))
            left, right = left + [left[-1]] * (n - len(left)), right + [right[-1]] * (n - len(right))
            for k in range(n):
                tags = []
                for name_, info, end in (("scripted (straight line)", li, ends[0]), ("RL (DDPG + HER)", ri, ends[1])):
                    tags.append((f"{name_}: {outcome(info)[0]}", outcome(info)[1]) if k >= end else (name_, (0, 0, 0)))
                frames.append(np.concatenate([label(left[k], *tags[0]), label(right[k], *tags[1])], axis=1))
            results.append(ri["is_success"])
            print(f"episode {i}: scripted {outcome(li)[0]}, RL {outcome(ri)[0]}", flush=True)
        else:
            ep, info = episode_frames(env, scripted if a.scripted else policy, TEST_SEED0 + i)
            frames += ep
            results.append(info["is_success"])
            print(f"episode {i}: success={info['is_success']}" + (" collision" if info.get("collision") else ""),
                  flush=True)
    name = f"{cfg['name']}_scripted" if a.scripted else run.name + ("_vs" if a.side_by_side else "")
    out = ROOT / "results" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    fps = (env.metadata.get("render_fps") or 25) / SLOWDOWN
    imageio.mimsave(out / f"{name}.mp4", frames, fps=fps, macro_block_size=8)
    small = [f[::2, ::2] for f in frames]  # half resolution keeps the README GIF small
    imageio.mimsave(out / f"{name}.gif", small, duration=1000 / fps, loop=0)
    print(f"{np.mean(results):.0%} success; wrote {out / name}.mp4/.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()

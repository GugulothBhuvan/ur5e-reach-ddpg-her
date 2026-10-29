"""Train DDPG/SAC + HER from a config. Resumes from runs/<name>_s<seed>/checkpoint if present.

python train.py --config configs/panda_ddpg.yaml --seed 0 [--steps 30000] [--threads 1]

Writes to runs/<name>_s<seed>/:
  train_log.csv   every 1000 steps: rollout success / final distance over the last 100 episodes
  eval.csv        every 2500 steps: 20 deterministic episodes (fixed goals)
  checkpoint/     model + replay buffer every 10k steps (overwritten)
  final_model.zip, done.json
"""
import argparse
import csv
import json
import shutil
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from common import algo_class, load_config, make_env, make_model, run_dir

LOG_EVERY, EVAL_EVERY, EVAL_EPISODES, CKPT_EVERY = 1000, 2500, 20, 10_000


def final_distance(obs):
    return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))


def run_episodes(model, env, n, seed0, deterministic=True):
    """n episodes on fixed goals (reset seeds seed0..seed0+n-1); returns (success, final distance) arrays."""
    succ, dist = [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        succ.append(float(info["is_success"]))
        dist.append(final_distance(obs))
    return np.array(succ), np.array(dist)


class Csv:
    def __init__(self, path, fields, keep_upto):
        """Append-mode CSV; on resume, drops rows logged after the checkpoint we resumed from."""
        rows = []
        if path.exists() and keep_upto is not None:
            with open(path) as f:
                rows = [r for r in csv.DictReader(f) if int(r["timesteps"]) <= keep_upto]
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.w.writerows(rows)
        self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()


class Logger(BaseCallback):
    def __init__(self, out, eval_env, wall0, episodes0, keep_upto):
        super().__init__()
        self.out, self.eval_env, self.wall0 = out, eval_env, wall0
        self.succ, self.dist = deque(maxlen=100), deque(maxlen=100)
        self.episodes = episodes0
        self.t0 = time.time()
        self.train_csv = Csv(out / "train_log.csv", ["timesteps", "wall_s", "episodes", "rollout_success",
                                                     "rollout_final_dist"], keep_upto)
        self.eval_csv = Csv(out / "eval.csv", ["timesteps", "wall_s", "eval_success", "eval_final_dist"], keep_upto)

    def wall(self):
        return self.wall0 + time.time() - self.t0

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes += 1
                self.succ.append(float(info["is_success"]))
                self.dist.append(final_distance(info["terminal_observation"]))
        t = self.num_timesteps
        if t % LOG_EVERY == 0 and self.succ:
            self.train_csv.write({"timesteps": t, "wall_s": round(self.wall(), 1), "episodes": self.episodes,
                                  "rollout_success": np.mean(self.succ), "rollout_final_dist": np.mean(self.dist)})
        if t % EVAL_EVERY == 0:
            s, d = run_episodes(self.model, self.eval_env, EVAL_EPISODES, seed0=1_000_000)
            self.eval_csv.write({"timesteps": t, "wall_s": round(self.wall(), 1), "eval_success": s.mean(),
                                 "eval_final_dist": d.mean()})
            print(f"{t:>7} steps  {self.wall() / 60:6.1f} min  rollout {np.mean(self.succ):.2f}  "
                  f"eval {s.mean():.2f}  dist {d.mean() * 100:.1f} cm", flush=True)
        if t % CKPT_EVERY == 0:
            tmp = self.out / "checkpoint.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            self.model.save(tmp / "model.zip")
            self.model.save_replay_buffer(tmp / "replay_buffer.pkl")
            (tmp / "state.json").write_text(json.dumps({"timesteps": t, "wall_s": self.wall(),
                                                            "episodes": self.episodes}))
            shutil.rmtree(self.out / "checkpoint", ignore_errors=True)
            tmp.rename(self.out / "checkpoint")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, help="override total_steps")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    cfg = load_config(a.config)
    total = a.steps or cfg["total_steps"]
    out = run_dir(cfg, a.seed)
    out.mkdir(parents=True, exist_ok=True)
    env, eval_env = make_env(cfg), make_env(cfg)

    ckpt = out / "checkpoint"
    if (ckpt / "model.zip").exists():
        state = json.loads((ckpt / "state.json").read_text())
        model = algo_class(cfg).load(ckpt / "model.zip", env=env)
        model.load_replay_buffer(ckpt / "replay_buffer.pkl")
        wall0, episodes0, keep_upto = state["wall_s"], state["episodes"], state["timesteps"]
        print(f"resumed {out.name} at {model.num_timesteps} steps", flush=True)
    else:
        model = make_model(cfg, env, a.seed)
        wall0, episodes0, keep_upto = 0.0, 0, None
        (out / "config.json").write_text(json.dumps(cfg, indent=2))

    remaining = total - model.num_timesteps
    if remaining > 0:
        cb = Logger(out, eval_env, wall0, episodes0, keep_upto)
        model.learn(remaining, callback=cb, reset_num_timesteps=False)
        wall = cb.wall()
    else:
        wall = wall0
    model.save(out / "final_model.zip")
    (out / "done.json").write_text(json.dumps({"timesteps": int(model.num_timesteps), "wall_s": round(wall, 1),
                                               "steps_per_s": round(model.num_timesteps / max(wall, 1e-9), 2)}))
    print(f"done {out.name}: {model.num_timesteps} steps in {wall / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()

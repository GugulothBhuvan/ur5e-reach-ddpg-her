"""N deterministic test episodes on goals never used in training or in-training eval.

python evaluate.py --config configs/panda_ddpg.yaml --seed 0 [--episodes 100] [--model final_model.zip]
Prints and writes results/<name>_s<seed>/test.json (success rate, mean / median final distance).
"""
import argparse
import json
import shutil

import numpy as np
import torch

from common import ROOT, algo_class, load_config, make_env, run_dir
from train import run_episodes

TEST_SEED0 = 2_000_000  # disjoint from training resets and from the in-training eval goals (1_000_000+)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--model", default="final_model.zip")
    a = p.parse_args()
    torch.set_num_threads(1)

    cfg = load_config(a.config)
    run = run_dir(cfg, a.seed)
    env = make_env(cfg)
    model = algo_class(cfg).load(run / a.model, env=env)
    succ, dist = run_episodes(model, env, a.episodes, seed0=TEST_SEED0)
    res = {"run": run.name, "model": a.model, "timesteps": int(model.num_timesteps), "episodes": a.episodes,
           "success_rate": float(succ.mean()), "mean_final_dist_m": float(dist.mean()),
           "median_final_dist_m": float(np.median(dist)), "max_final_dist_m": float(dist.max())}
    done = run / "done.json"
    if done.exists():
        res["train_wall_s"] = json.loads(done.read_text())["wall_s"]
    out = ROOT / "results" / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "test.json").write_text(json.dumps(res, indent=2))
    for f in ("train_log.csv", "eval.csv", "done.json", "config.json"):  # small logs kept in git
        if (run / f).exists():
            shutil.copy(run / f, out / f)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

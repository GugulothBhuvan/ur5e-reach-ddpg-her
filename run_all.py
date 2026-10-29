"""Run several training jobs in parallel, then evaluate, record and plot. Same on a laptop, Kaggle, RunPod.

python run_all.py --jobs panda_ddpg:0,1,2 ur5e_ddpg:0 --steps 100000 --parallel 4 [--hours 11.3] [--no-video]

- jobs are config:seeds; each trains with train.py (resumes from runs/<name>_s<seed>/checkpoint)
- on a GPU machine jobs are spread over the visible GPUs (several per GPU is fine: the nets are small)
- --hours: stop training before this wall-clock budget (e.g. Kaggle's 12 h limit); rerun to continue
- afterwards: evaluate.py (100 test episodes) and record.py (10-episode MP4/GIF) for every finished run,
  plot.py per config, and a results table at results/summary.md
"""
import argparse
import json
import os
import subprocess
import sys
import time

from common import ROOT, RUNS, load_config

PY = sys.executable


def gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def finished(name, seed, steps):
    done = RUNS / f"{name}_s{seed}" / "done.json"
    return done.exists() and json.loads(done.read_text())["timesteps"] >= steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", nargs="+", required=True, help="config:seed[,seed...], e.g. panda_ddpg:0,1,2")
    p.add_argument("--steps", type=int, help="override every config's total_steps")
    p.add_argument("--parallel", type=int, default=os.cpu_count())
    p.add_argument("--hours", type=float, help="training wall-clock budget")
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    deadline = t0 + a.hours * 3600 if a.hours else None

    jobs = [(name, int(s)) for j in a.jobs for name, seeds in [j.split(":")] for s in seeds.split(",")]
    steps = {name: a.steps or load_config(ROOT / "configs" / f"{name}.yaml")["total_steps"] for name, _ in jobs}
    todo = [j for j in jobs if not finished(*j, steps[j[0]])]
    n_gpu = gpus()
    print(f"{len(jobs)} jobs, {len(todo)} to train, {a.parallel} in parallel, {n_gpu} GPU(s)", flush=True)

    (ROOT / "logs").mkdir(exist_ok=True)
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            name, seed = todo.pop(0)
            env = dict(os.environ)
            if n_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(len(running) % n_gpu)
            cmd = [PY, "train.py", "--config", f"configs/{name}.yaml", "--seed", str(seed),
                   "--steps", str(steps[name]), "--threads", "1"]
            log = open(ROOT / "logs" / f"{name}_s{seed}.log", "a")
            running.append((name, seed, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
            print(f"started {name} seed {seed}", flush=True)
        time.sleep(30)
        for job in list(running):
            if job[2].poll() is not None:
                running.remove(job)
                print(f"{job[0]} seed {job[1]} exited ({job[2].returncode})", flush=True)
        if deadline and time.time() > deadline and running:
            print("time budget reached: stopping (checkpoints every 10k steps; rerun to resume)", flush=True)
            for _, _, proc in running:
                proc.terminate()
            for _, _, proc in running:
                proc.wait()
            break

    done = [j for j in jobs if finished(*j, steps[j[0]])]
    print(f"{len(done)}/{len(jobs)} runs finished", flush=True)
    for name, seed in done:
        args = ["--config", f"configs/{name}.yaml", "--seed", str(seed)]
        subprocess.run([PY, "evaluate.py", *args], cwd=ROOT, check=True)
        if not a.no_video:
            subprocess.run([PY, "record.py", *args], cwd=ROOT, check=True)
    names = sorted({n for n, _ in done})
    if names:
        subprocess.run([PY, "plot.py", *names], cwd=ROOT, check=True)
        summary(done)


def summary(done):
    rows = ["| run | steps | test success (100 ep) | mean final distance | train wall-clock |", "|---|---|---|---|---|"]
    for name, seed in done:
        r = json.loads((ROOT / "results" / f"{name}_s{seed}" / "test.json").read_text())
        rows.append(f"| {r['run']} | {r['timesteps']:,} | {r['success_rate']:.0%} | "
                    f"{r['mean_final_dist_m'] * 100:.1f} cm | {r.get('train_wall_s', 0) / 3600:.2f} h |")
    (ROOT / "results" / "summary.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

"""Watchdog / job queue for long training sessions: keeps N runs going, restarts crashed or stalled runs.

python watchdog.py [--queue logs/queue.txt] [--parallel 4] [--every 60] [--stall-min 10]

The queue file has one job per line, `config seed steps` (e.g. `ur5e_c2_ee 1 100000`); lines can be added
while the watchdog runs. Every --every seconds it:
  - marks a job done when runs/<name>_s<seed>/done.json reports >= steps
  - leaves a running job alone, unless its log has not grown for --stall-min minutes (then kills it)
  - (re)starts unfinished jobs while fewer than --parallel are running; train.py resumes from the checkpoint
  - writes logs/watchdog_status.md (one row per job: state, last log line, restarts)
It exits when every job in the queue is done.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from common import ROOT, RUNS


def read_queue(path):
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, seed, steps = line.split()
            jobs.append((name, int(seed), int(steps)))
    return jobs


def done(name, seed, steps):
    f = RUNS / f"{name}_s{seed}" / "done.json"
    return f.exists() and json.loads(f.read_text())["timesteps"] >= steps


def running_pid(name, seed):
    """PID of a train.py process for this job (started by us or by hand), else None."""
    out = subprocess.run(["pgrep", "-f", f"train.py --config configs/{name}.yaml --seed {seed}( |$)"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def last_line(log):
    if not log.exists():
        return ""
    lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", default="logs/queue.txt")
    p.add_argument("--parallel", type=int, default=os.cpu_count())
    p.add_argument("--every", type=int, default=60)
    p.add_argument("--stall-min", type=float, default=10)
    a = p.parse_args()
    queue, logs = ROOT / a.queue, ROOT / "logs"
    logs.mkdir(exist_ok=True)
    restarts = {}
    while True:
        jobs = read_queue(queue)
        rows, n_running, pending = [], 0, []
        for name, seed, steps in jobs:
            log = logs / f"{name}_s{seed}.log"
            pid = running_pid(name, seed)
            if done(name, seed, steps) and pid is None:
                state = "done"
            elif pid is not None:
                stalled = log.exists() and time.time() - log.stat().st_mtime > a.stall_min * 60
                if stalled:
                    os.kill(pid, signal.SIGKILL)
                    state = "stalled, killed (will resume)"
                else:
                    state, n_running = "running", n_running + 1
            else:
                state = "waiting"
                pending.append((name, seed, steps))
            rows.append([name, seed, steps, state, last_line(log), restarts.get((name, seed), 0)])
        for name, seed, steps in pending:
            if n_running >= a.parallel:
                break
            key = (name, seed)
            if (RUNS / f"{name}_s{seed}").exists():
                restarts[key] = restarts.get(key, 0) + 1
            cmd = [sys.executable, "train.py", "--config", f"configs/{name}.yaml", "--seed", str(seed),
                   "--steps", str(steps), "--threads", "1"]
            with open(logs / f"{name}_s{seed}.log", "a") as f:
                subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            n_running += 1
            for r in rows:
                if r[0] == name and r[1] == seed:
                    r[3], r[5] = "started", restarts.get(key, 0)
        table = ["| job | steps | state | last log line | (re)starts |", "|---|---|---|---|---|"]
        table += [f"| {n}_s{s} | {st:,} | {state} | `{line[:90]}` | {rs} |" for n, s, st, state, line, rs in rows]
        (logs / "watchdog_status.md").write_text(f"updated {time.strftime('%H:%M:%S')}\n\n" + "\n".join(table) + "\n")
        if all(r[3] == "done" for r in rows):
            print("all jobs done", flush=True)
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()

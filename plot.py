"""Learning curves from runs/<name>_s*/ (train_log.csv, eval.csv) -> results/plots/.

python plot.py panda_ddpg ur5e_ddpg [--out learning_curves.png]
Each config gets one panel: rollout success (last 100 training episodes, as the tutorial logged) and
deterministic eval success (20 fixed goals), mean over seeds with min-max band. For panda_ddpg the
tutorial's logged rollout success is overlaid.
"""
import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT, RUNS  # noqa: E402

TUTORIAL = {"steps": [5_000, 7_000, 100_000, 1_000_000], "success": [0.70, 0.95, 0.96, 1.00]}
TITLES = {"panda_ddpg": "PandaReach-v3, DDPG + HER", "panda_ddpg_1critic": "PandaReach-v3, pure DDPG (1 critic) + HER",
          "ur5e_ddpg": "UR5e + 2F-85 reach, DDPG + HER", "ur5e_sac": "UR5e + 2F-85 reach, SAC + HER",
          "ur5e_ddpg_2cm": "UR5e + 2F-85 reach, DDPG + HER, 2 cm success"}


def band(ax, dfs, col, label, color):
    df = pd.concat([d.set_index("timesteps")[col].rename(i) for i, d in enumerate(dfs)], axis=1).sort_index()
    ax.plot(df.index, df.mean(axis=1), color=color, label=f"{label} (mean of {len(dfs)} seed{'s' * (len(dfs) > 1)})")
    if len(dfs) > 1:
        ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), color=color, alpha=0.2, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--out", default="learning_curves.png")
    a = p.parse_args()
    fig, axes = plt.subplots(1, len(a.names), figsize=(6.2 * len(a.names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], a.names):
        runs = sorted(RUNS.glob(f"{name}_s[0-9]*"))
        train = [pd.read_csv(r / "train_log.csv") for r in runs]
        evals = [pd.read_csv(r / "eval.csv") for r in runs]
        band(ax, train, "rollout_success", "rollout success (last 100 episodes)", "#2a6fdb")
        band(ax, evals, "eval_success", "deterministic eval (20 goals)", "#e07b00")
        if name.startswith("panda"):
            ax.plot(TUTORIAL["steps"], TUTORIAL["success"], "k*", ms=11, label="tutorial log (rollout, GPU run)")
        xmax = max(d["timesteps"].max() for d in train)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.95 if name.startswith("panda") else 0.90, color="grey", ls=":", lw=1)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("success rate")
        ax.set_title(TITLES.get(name, name))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = ROOT / "results" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / a.out, dpi=130)
    print("wrote", out / a.out)


if __name__ == "__main__":
    main()

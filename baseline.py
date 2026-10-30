"""Scripted controller, no learning: drive straight at the goal. Same 100 test episodes as evaluate.py.

python baseline.py --config configs/ur5e_ddpg.yaml [--episodes 100]
Writes results/<name>/scripted.json (success rate, mean / median / max final distance, mean episode steps,
collision rate). The rule:
  gripper (IK) mode   action = clip((goal - tcp) / 0.05, -1, 1); with an orientation goal the rotation part is
                      clip(rotation vector from current to target orientation / 0.1 rad, -1, 1)
  joint mode          one solve of the env's IK (holds gripper-down) toward a point 5 cm closer to the goal,
                      turned into a joint increment: clip((q_ik - q) / 0.05, -1, 1). --position-only solves
                      for position alone (a stronger variant; results/<name>/scripted_posonly.json)
  obstacles           ignored: it is the straight-line rule
"""
import argparse
import json

import mujoco
import numpy as np

from common import ROOT, load_config, make_env
from evaluate import TEST_SEED0


def scripted_action(env, obs, position_only=False):
    """env: the UR5eReachEnv (unwrapped); obs: its observation dict."""
    if not hasattr(env, "tcp_pos"):  # panda-gym PandaReach (gripper actions, 5 cm per unit)
        return np.clip((obs["desired_goal"] - obs["achieved_goal"]) / 0.05, -1, 1).astype(np.float32)
    goal, tcp = np.asarray(obs["desired_goal"], dtype=np.float64), env.tcp_pos()
    delta = goal[:3] - tcp
    if env.control_type == "ee":
        a = np.clip(delta / env.action_scale, -1, 1)
        if env.orientation_goal:
            cur, neg, err, rv = env.tcp_quat_now(), np.empty(4), np.empty(4), np.empty(3)
            mujoco.mju_negQuat(neg, cur)
            mujoco.mju_mulQuat(err, env.goal_quat, neg)  # world-frame rotation from current to target
            mujoco.mju_quat2Vel(rv, err, 1.0)
            a = np.concatenate([a, np.clip(rv / env.rot_scale, -1, 1)])
        return a.astype(np.float32)
    dist = np.linalg.norm(delta)
    target = goal[:3] if dist <= 0.05 else tcp + delta / dist * 0.05
    q = env.data.qpos[env.qadr].copy()
    return np.clip((env.ik(target, q, orient=not position_only) - q) / 0.05, -1, 1).astype(np.float32)


def run_scripted(env, n, seed0, position_only=False):
    """Like train.run_episodes, for the scripted rule. Returns per-episode success, distance, steps, collision."""
    inner = env.unwrapped
    succ, dist, steps, coll = [], [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done, t = False, 0
        while not done:
            obs, _, terminated, truncated, info = env.step(scripted_action(inner, obs, position_only))
            done, t = terminated or truncated, t + 1
        succ.append(float(info["is_success"]))
        dist.append(float(np.linalg.norm(obs["achieved_goal"][:3] - obs["desired_goal"][:3])))
        steps.append(t)
        coll.append(float(info.get("collision", False)))
    return np.array(succ), np.array(dist), np.array(steps), np.array(coll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--position-only", action="store_true", help="joint mode: position-only IK")
    a = p.parse_args()
    cfg = load_config(a.config)
    env = make_env(cfg)
    succ, dist, steps, coll = run_scripted(env, a.episodes, TEST_SEED0, a.position_only)
    res = {"config": cfg["name"], "controller": "scripted (straight at the goal, no learning)"
           + (", position-only IK" if a.position_only else ""),
           "episodes": a.episodes, "success_rate": float(succ.mean()), "mean_final_dist_m": float(dist.mean()),
           "median_final_dist_m": float(np.median(dist)), "max_final_dist_m": float(dist.max()),
           "mean_steps": float(steps.mean()), "mean_steps_successful": float(steps[succ > 0].mean()) if succ.any() else None}
    if getattr(env.unwrapped, "obstacle", False):
        res["collision_rate"] = float(coll.mean())
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / ("scripted_posonly.json" if a.position_only else "scripted.json")).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

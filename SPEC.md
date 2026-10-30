# SPEC: basic reach demo, DDPG + HER (Panda, then UR5e)

MTP (IIT Kharagpur), "basic version first" requested by the guide. Standalone; unrelated to the ResPlan
residual-RL project.

## Phase A: replicate the tutorial (panda-gym PandaReach, DDPG + HER)

Tutorial (panda-gym 2.0.0, gym, SB3 1.5, Python 3.7, GPU):
- `PandaReach-v2` + `TimeFeatureWrapper`; sparse reward (-1 / 0), 50-step episodes, success at 5 cm
- `DDPG("MultiInputPolicy", replay_buffer_class=HerReplayBuffer)`, HER `future`, `n_sampled_goal=4`,
  `online_sampling=True`
- `gamma=0.95`, `batch_size=2048`, `buffer_size=100_000`, `learning_rate=1e-3`,
  `net_arch=[512, 512, 512]`, `n_critics=2`, `NormalActionNoise(sigma=0.1)`, 1e6 steps
- logged rollout success: ~70 % at 5k steps, ~95 % at 7k, 96-100 % from ~100k; ~5.8 h at ~48 steps/s

Port: panda-gym 3.x (`PandaReach-v3`, gymnasium), SB3 2.x, sb3-contrib, CPU torch. Every change goes in
the README. Same hyperparameters. Steps sized from measured speed (target 100k; the curve plateaus well
before). Up to 3 seeds. Optional: `n_critics=1` (pure DDPG; `n_critics=2` is TD3's clipped double-Q).

Outputs: 100 deterministic test episodes (success rate, mean final distance), learning curve against the
tutorial's logged points, MP4 of ~10 episodes. Notebook updated to run on current libraries.
**Acceptance: >= 95 % test success.**

## Phase B: same recipe on UR5e + Robotiq 2F-85 (MuJoCo)

`envs/ur5e_reach.py`, gymnasium goal env, registered as `UR5eReach-v0` (TimeLimit 50):
- Menagerie UR5e, 2F-85 on the flange, fingers held open; TCP site between the fingertips
- obs dict as PandaReach: `observation` = TCP pos + TCP vel (6), `achieved_goal` = TCP pos,
  `desired_goal` = target
- action: 3 in [-1, 1] -> TCP displacement x 5 cm; damped least-squares IK (`mj_jacSite`, position +
  hold gripper-down orientation, as panda-gym's IK holds the end-effector orientation) -> clip to joint
  limits -> position actuators -> 20 physics steps (0.04 s, panda-gym's control period)
- start: fixed ready pose, TCP at the goal-cube centre, gripper pointing down
- goal: uniform in a 0.3 m cube in front of the robot; reachability check: IK controller driven straight
  at 500 goals must reach >= 99 %
- reward: sparse, -1 if d > 5 cm else 0; terminated on success (as panda-gym 3); `compute_reward` vectorised
  for HER
- rendering: `rgb_array`, offscreen, fixed camera, translucent visual-only goal sphere
- tests: IK direction, `compute_reward` == `step()` reward, `check_env`, 1000 random steps without NaN

Train with the identical DDPG + HER settings + `TimeFeatureWrapper`; evaluate, record and plot as Phase A.
**Acceptance: >= 90 % test success at 5 cm** (report honestly if short).

Stretch, in order, after the core: SAC + HER; 2 cm threshold; joint-space actions.

## Phase C: harder tasks, scripted baseline vs RL

A scripted controller with no learning ("move straight at the goal") gets 100 % on every Phase B setting, so
Phase B does not show that RL is needed. Phase C makes the UR5e task harder one change at a time and reports
the scripted rule next to RL on the same 100 test episodes (reset seeds 2,000,000+) at every stage.

- Phases A and B stay as they are. New env features are constructor flags whose defaults reproduce Phase B
  bit for bit (checked step by step against the Phase B env on all three Phase B configs).
- Light training settings: `net_arch=[256, 256]`, `batch_size=256`, otherwise as Phase B (DDPG + HER `future`
  x4, `n_critics=2`, gamma 0.95, sigma 0.1, lr 1e-3, buffer 100k). 30k steps, up to 100k if still rising.
  1 seed while developing, 3 for the final number where time allows.
- `baseline.py --config`: the scripted rule on the 100 test episodes -> `results/<name>/scripted.json`.
  Gripper mode `clip((goal - tcp) / 0.05, -1, 1)`; joint mode: one solve of the env's IK toward a point 5 cm
  closer, as a joint increment; from C3 it also turns toward the target orientation; from C4 it ignores the
  obstacle.
- Metrics: success rate, mean final distance, collision rate (C4).

| Stage | Change (cumulative from C2) | Bar |
|---|---|---|
| C0 | scripted baseline in the repo; "Scripted" column for Phase B | reproduce 100 % / 2.7 steps etc. |
| C1 | Phase B task, light settings | RL 100 %; report the speedup |
| C2 | `random_start` (rejection-sampled joints: within limits, TCP >= 5 cm up, no self/floor contact), `observe_joints` (sin, cos, q / limit: +18), goal cube 0.5 m (centre raised to z = 0.30 so it clears the floor); gripper and joint mode | RL >= 90 % (gripper); joint mode reported |
| C3 | `orientation_goal`: goal + 6D orientation (9 numbers), success = 5 cm AND 15 deg, targets within 45 deg of gripper-down; gripper action 6 numbers (5 cm, 0.1 rad), IK tracks the full pose; batched `compute_reward` | RL >= 80 % |
| C4 | `obstacle`: one sphere (r 5-8 cm) near the start->goal midpoint (+-3 cm), clear of start pose and goal; obs + centre and radius (+4); contact ends the episode with reward -1 / (1 - gamma) = -20; success = reached and no collision; HER keeps the penalty (`info["collision"]`, `copy_info_dict=True`) | RL success > scripted success |

Stretch after C4: 2-3 obstacles; SAC vs DDPG on C4; side-by-side video scripted vs RL.

Added during Phase C (diagnostics, same light settings, 1 seed): C3 and C4 from the fixed Phase B start
(`*_fixedstart`) to separate orientation / obstacle difficulty from random-start difficulty; a hand-coded detour
(`baseline.py --detour`: up over the spheres, across, down) to show the obstacle task is solvable; `watchdog.py`
(job queue, restarts crashed / stalled runs). Seeds: 3 for stages that pass their bar, 1 for stages that miss.

Outcome (README "Phase C"): C1 passes (100 %, 3 seeds). RL beats the scripted rule from C2 (gripper: 77 % vs
69 %) but misses the 90 % bar; C2 joints, C3 and C4 miss their bars. From the fixed start RL learns the
orientation goal (100 %) and to avoid obstacles, but not to go around them (detour rule: 77 %).

## Compute (measured)

Container: 4 CPUs, no GPU. With the tutorial's settings (batch 2048, 3x512 twin critics, 1 gradient step
per env step) training runs at ~3 steps/s on 1 thread, ~5.3 on 2, ~4.7 on 4 (env stepping is ~500-1200
steps/s, so the networks dominate). Best throughput: 4 independent 1-thread runs.
Phase C light settings (2x256 critics, batch 256): ~46 steps/s on 1 thread vs ~2.4 for the tutorial settings
on the same machine (~19x); 30k steps take ~11 min. Plan: Panda seeds 0-2
and UR5e seed 0 in parallel to 30k steps (~2.8 h), then resume to 100k. Checkpoints (model + replay
buffer) every 10k steps; runs resume from them.

## Layout

```
configs/            one YAML per run
envs/ur5e_reach.py  Phase B env
assets/             Menagerie UR5e + 2F-85 (licences kept)
train.py            --config --seed; checkpoints + CSV logs
evaluate.py         N deterministic test episodes
record.py           MP4/GIF of the trained policy
plot.py             learning curves
notebooks/          updated tutorial notebook
results/            small CSVs, plots, videos
```

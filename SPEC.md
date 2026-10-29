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

## Compute (measured)

Container: 4 CPUs, no GPU. With the tutorial's settings (batch 2048, 3x512 twin critics, 1 gradient step
per env step) training runs at ~3 steps/s on 1 thread, ~5.3 on 2, ~4.7 on 4 (env stepping is ~500-1200
steps/s, so the networks dominate). Best throughput: 4 independent 1-thread runs. Plan: Panda seeds 0-2
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

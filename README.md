# ur5e-reach-ddpg-her

Basic reach demo for my MTP (RL trajectory planning, IIT Kharagpur), in two phases:

- **Phase A:** replicate a YouTube tutorial, DDPG + HER on panda-gym's `PandaReach`, on today's libraries.
- **Phase B:** the identical DDPG + HER recipe on a **UR5e + Robotiq 2F-85** reach task in MuJoCo
  (`envs/ur5e_reach.py`).

Design notes are in [SPEC.md](SPEC.md).

## Results

RESULTS_TABLE

| Panda (`PandaReach-v3`) | UR5e + 2F-85 (`UR5eReach-v0`) |
|---|---|
| ![panda](results/videos/panda_ddpg_s0.gif) | ![ur5e](results/videos/ur5e_ddpg_s0.gif) |

MP4s are in [results/videos/](results/videos/). Learning curves:

![learning curves](results/plots/learning_curves.png)

## Run

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest tests                                          # UR5e env checks

python train.py --config configs/panda_ddpg.yaml --seed 0       # resumes from runs/<name>_s<seed>/checkpoint
python evaluate.py --config configs/panda_ddpg.yaml --seed 0    # 100 deterministic test episodes
python record.py --config configs/panda_ddpg.yaml --seed 0      # results/videos/*.mp4 + .gif
python plot.py panda_ddpg ur5e_ddpg                             # results/plots/learning_curves.png
```

Swap in `configs/ur5e_ddpg.yaml` for Phase B. `train.py --steps N` overrides the config's step budget.
Rendering is headless (MuJoCo via OSMesa: `apt install libosmesa6`; panda-gym via PyBullet's TinyRenderer).
The updated tutorial notebook is [notebooks/panda_reach_ddpg_her.ipynb](notebooks/panda_reach_ddpg_her.ipynb).

## Differences from the tutorial

DIFFERENCES

## UR5e reach environment (Phase B)

Mirrors `PandaReach` as closely as the robots allow:

| | PandaReach-v3 | UR5eReach-v0 |
|---|---|---|
| Robot | Franka Panda (7 joints), PyBullet | UR5e (6 joints) + Robotiq 2F-85 held open, MuJoCo Menagerie models |
| Observation | EE position + velocity (6) | TCP (point between the fingertips) position + velocity (6) |
| Goals | `achieved_goal` EE position, `desired_goal` target | same, with the TCP |
| Action | 3 in [-1, 1] x 5 cm EE displacement -> IK (orientation held) -> joint position control | same: damped least-squares IK with `mj_jacSite` on position + gripper-down orientation, clipped to joint limits, position actuators, 20 physics steps (0.04 s) |
| Goal region | 0.3 m cube | 0.3 m cube centred 0.45 m in front of the base, 0.25 m up |
| Reward / success | sparse -1 / 0, success at 5 cm, terminates on success, 50 steps | same |

UR5e-specific choices, all routine:
- **Ready pose:** TCP at the goal-cube centre, gripper pointing straight down.
- **Gravity compensation** on the arm and gripper bodies, as a real UR controller provides. Without it the
  position servos settle about 1 cm short of the IK target.
- **Arm servo stiffness 5x Menagerie's** (damping and the real 150 / 28 N·m torque limits unchanged). At stock
  gains a full action moves the TCP only 0.9 cm per step, against panda-gym's 2.5-5.8 cm. At 5x it moves
  about 3 cm (about 0.78 m/s, under the UR5e's 1 m/s rated tool speed).
- **Reachability check:** the IK controller driven straight at the goal reaches 500 / 500 sampled goals,
  converging to within 0.1 mm. This is checked in `tests/`.

Tests in `tests/test_env.py` check that:
- IK moves the TCP in the commanded direction;
- the gripper stays pointing down;
- `compute_reward` matches `step()`, including vectorised for HER;
- gymnasium's `check_env` passes;
- 1,000 random steps produce no NaNs;
- 500 goals are reachable.

## Layout

```
configs/            one YAML per run
envs/ur5e_reach.py  Phase B gymnasium env (registered as UR5eReach-v0)
assets/             MuJoCo Menagerie UR5e + Robotiq 2F-85 (BSD licences inside; commit in MENAGERIE_COMMIT)
train.py            --config --seed; checkpoints (model + replay buffer) every 10k steps, CSV logs
evaluate.py         100 deterministic test episodes -> results/<run>/test.json
record.py           MP4/GIF of the trained policy
plot.py             learning curves
notebooks/          updated tutorial notebook
results/            CSV logs, test results, plots, videos
```

"""UR5e + Robotiq 2F-85 reach task in MuJoCo, mirroring panda-gym's PandaReach.

Same interface as PandaReach-v3:
  observation    TCP position (3) + TCP linear velocity (3)
  achieved_goal  TCP position (3)
  desired_goal   target position (3)
  action         3 numbers in [-1, 1] -> TCP target displacement of up to 5 cm per step, converted to
                 joint targets by damped least-squares IK (gripper held pointing down), sent to the
                 position actuators, then 20 physics steps (0.04 s) to settle (the arm covers ~3 cm)
  control_type   "ee" (default, above) or "joints": 6 numbers in [-1, 1] -> joint-angle increments of up to
                 0.05 rad added to the current angles (panda-gym's "joints" mode; observation unchanged)
  reward         sparse: -1 while the TCP is more than 5 cm from the goal, 0 otherwise
  episode        terminated on success, truncated at 50 steps (TimeLimit, set at registration)
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ASSETS = Path(__file__).resolve().parent.parent / "assets"

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
# Arm servo stiffness relative to Menagerie's (kp 2000 / 500, kv unchanged, torque limits unchanged). At the
# stock gains a full action moves the TCP 0.9 cm per control step; at 5x it moves ~3 cm, close to the
# commanded 5 cm and to panda-gym's Panda (2.5-5.8 cm), and still within the UR5e's 150 / 28 Nm limits.
KP_SCALE = 5.0
# Ready pose: TCP at the goal-cube centre, gripper pointing straight down (found once with the IK
# below from the Menagerie "home" keyframe; tests/test_env.py checks both properties).
READY_QPOS = np.array([-1.8732, -1.6993, 1.9015, -1.7730, -1.5708, -0.3024])


CAM_POS, CAM_TARGET = np.array([1.15, 1.25, 0.85]), np.array([0.0, 0.35, 0.25])


def _look_at(pos, target):
    """MuJoCo camera xyaxes for a camera at pos looking at target, z up."""
    fwd = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(fwd, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.concatenate([x, np.cross(x, fwd)]).tolist()


def build_model() -> mujoco.MjModel:
    """Menagerie UR5e with the 2F-85 on its flange, a floor, a camera and a visual-only goal sphere."""
    spec = mujoco.MjSpec.from_file(str(ASSETS / "universal_robots_ur5e" / "ur5e.xml"))
    gripper = mujoco.MjSpec.from_file(str(ASSETS / "robotiq_2f85" / "2f85.xml"))
    # TCP: the point between the fingertips (the 2F-85's own "pinch" site sits there too)
    gripper.body("base").add_site(name="tcp", pos=[0, 0, 0.145], size=[0.004], rgba=[1, 0, 0, 0],
                                  group=5)
    spec.option.cone, spec.option.impratio = gripper.option.cone, gripper.option.impratio  # 2F-85's settings
    spec.site("attachment_site").attach_body(gripper.body("base_mount"), "gripper/", "")

    for body in spec.bodies:  # gravity compensation, as a real UR controller provides
        if body.name not in ("world", "goal"):
            body.gravcomp = 1.0
    wb = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.84, 0.86], rgb2=[0.74, 0.76, 0.78], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.05)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], material="grid")
    wb.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.2, 0.2, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6])
    wb.add_camera(name="side", pos=CAM_POS, xyaxes=_look_at(CAM_POS, CAM_TARGET))
    goal = wb.add_body(name="goal", mocap=True, pos=[0.0, 0.45, 0.25])
    goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=[0.1, 0.8, 0.2, 0.45],
                  contype=0, conaffinity=0)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    model = spec.compile()
    arm = [model.actuator(n).id for n in ARM_ACTUATORS]
    model.actuator_gainprm[arm, 0] *= KP_SCALE
    model.actuator_biasprm[arm, 1] *= KP_SCALE
    return model


class UR5eReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None, distance_threshold=0.05, reward_type="sparse", control_type="ee",
                 goal_center=(0.0, 0.45, 0.25), goal_range=0.3, action_scale=0.05, n_substeps=20,
                 width=480, height=360):
        self.render_mode = render_mode
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.control_type = control_type
        self.n_substeps = n_substeps
        c, h = np.asarray(goal_center, dtype=np.float64), goal_range / 2
        self.goal_low, self.goal_high = c - h, c + h

        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)  # scratch copy for IK iterations
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.qlim = np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_act = np.array([m.actuator(n).id for n in ARM_ACTUATORS])
        self.tcp = m.site("gripper/tcp").id
        self.goal_mocap = m.body_mocapid[m.body("goal").id]
        # target TCP orientation: whatever the ready pose has (pointing down); IK holds it
        self._set_qpos(self.data, READY_QPOS)
        self.tcp_quat = np.empty(4)
        mujoco.mju_mat2Quat(self.tcp_quat, self.data.site_xmat[self.tcp])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3 if control_type == "ee" else 6,), dtype=np.float32)
        box = lambda n: spaces.Box(-10.0, 10.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"observation": box(6), "achieved_goal": box(3), "desired_goal": box(3)})
        self.goal = np.zeros(3)
        self._renderer = None
        self.width, self.height = width, height

    # ---------------------------------------------------------------- kinematics
    def _set_qpos(self, d, q):
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def ik(self, target_pos, q0, iters=20, damping=0.05, tol=1e-4):
        """Damped least squares on the 6-D TCP error (position + hold orientation)."""
        d, m = self._ik_data, self.model
        q = q0.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err, cur, neg = np.zeros(6), np.empty(4), np.empty(4)
        for _ in range(iters):
            self._set_qpos(d, q)
            err[:3] = target_pos - d.site_xpos[self.tcp]
            mujoco.mju_mat2Quat(cur, d.site_xmat[self.tcp])
            mujoco.mju_negQuat(neg, cur)
            dq_ = np.empty(4)
            mujoco.mju_mulQuat(dq_, self.tcp_quat, neg)
            mujoco.mju_quat2Vel(err[3:], dq_, 1.0)
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = np.vstack([jacp, jacr])[:, self.dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            q = np.clip(q + dq, self.qlim[:, 0], self.qlim[:, 1])
        return q

    # ---------------------------------------------------------------- gym API
    def tcp_pos(self):
        return self.data.site_xpos[self.tcp].copy()

    def _tcp_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tcp, v, 0)
        return v[3:]  # world-frame linear velocity

    def _obs(self):
        p = self.tcp_pos()
        return {"observation": np.concatenate([p, self._tcp_vel()]).astype(np.float32),
                "achieved_goal": p.astype(np.float32), "desired_goal": self.goal.astype(np.float32)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = READY_QPOS
        self.data.ctrl[self.arm_act] = READY_QPOS
        self.goal = self.np_random.uniform(self.goal_low, self.goal_high)
        self.data.mocap_pos[self.goal_mocap] = self.goal
        mujoco.mj_forward(self.model, self.data)
        obs = self._obs()
        return obs, {"is_success": self._is_success(obs["achieved_goal"], self.goal)}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q = self.data.qpos[self.qadr].copy()
        if self.control_type == "ee":
            target = self.tcp_pos() + action * self.action_scale
            target[2] = max(target[2], 0.02)  # keep the fingertips off the floor (panda-gym clips z >= 0)
            q_target = self.ik(target, q)
        else:
            q_target = np.clip(q + action * 0.05, self.qlim[:, 0], self.qlim[:, 1])
        self.data.ctrl[self.arm_act] = q_target
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        obs = self._obs()
        success = self._is_success(obs["achieved_goal"], self.goal)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))
        return obs, reward, success, False, {"is_success": success}

    def _is_success(self, achieved, desired):
        return bool(np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1) < self.distance_threshold)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        return -d.astype(np.float32)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

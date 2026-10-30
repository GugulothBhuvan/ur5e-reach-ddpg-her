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

Phase C options (all off by default, which reproduces Phase B exactly):
  random_start      start from a random joint configuration (rejection-sampled: within limits, TCP >= 5 cm
                    above the floor, no self- or floor contact) instead of the fixed ready pose
  observe_joints    append sin, cos and the limit-scaled angle of the 6 arm joints to `observation` (+18)
  orientation_goal  goals also carry a gripper orientation (6D: first two columns of the rotation matrix), so
                    achieved/desired goals are 9 numbers; success needs position within distance_threshold AND
                    angle within angle_threshold_deg; targets lie within goal_max_tilt_deg of gripper-down.
                    In "ee" mode the action becomes 6 numbers: TCP displacement (x 5 cm) + world-frame rotation
                    vector (x rot_scale rad), and IK tracks the full pose
  obstacle          one sphere (radius obstacle_radius) near the midpoint of the start-TCP -> goal line, offset
                    by up to obstacle_offset per axis; `observation` gains its centre and radius (+4). Any
                    contact between the robot and the sphere ends the episode with reward -collision_penalty
                    (-1 / (1 - gamma) = -20 at gamma 0.95) and counts as failure. The flag is returned as
                    info["collision"] and compute_reward reads it, so HER relabelling keeps the penalty
                    (use HerReplayBuffer(copy_info_dict=True))
  n_obstacles       with obstacle: number of spheres (default 1). Sphere 0 is placed as above; the others near
                    the points at 0.3 and 0.7 of the start-TCP -> goal line, same radius range and clearances, not
                    overlapping each other. Their per-axis offset starts at obstacle_offset and widens over the
                    retries to obstacle_spread, so they sit on the line when it is long enough and beside sphere 0
                    when not (e.g. from the fixed start). `observation` gains 4 numbers per sphere; touching any
                    sphere is a collision
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


def rotation_angle(a6, b6):
    """Angle (rad) between rotations given in 6D form (first two matrix columns); works on batches."""
    a6, b6 = np.asarray(a6, dtype=np.float64), np.asarray(b6, dtype=np.float64)

    def frame(x):  # Gram-Schmidt, so float32-stored goals still give proper rotations
        c1 = x[..., :3] / np.linalg.norm(x[..., :3], axis=-1, keepdims=True)
        c2 = x[..., 3:] - (c1 * x[..., 3:]).sum(-1, keepdims=True) * c1
        c2 /= np.linalg.norm(c2, axis=-1, keepdims=True)
        return c1, c2, np.cross(c1, c2)

    trace = sum((u * v).sum(-1) for u, v in zip(frame(a6), frame(b6)))  # trace(Ra^T Rb)
    return np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))


def collision_flags(info, shape):
    """info["collision"] from one info dict or a batch of them (as HerReplayBuffer passes with copy_info_dict)."""
    if isinstance(info, dict):
        return np.full(shape, bool(info.get("collision", False)))
    return np.array([bool(i.get("collision", False)) for i in info]).reshape(shape)


CAM_POS, CAM_TARGET = np.array([0.85, 1.05, 0.65]), np.array([0.0, 0.40, 0.22])
WIDE_POS, WIDE_TARGET = np.array([1.45, 1.55, 1.15]), np.array([0.0, 0.15, 0.35])  # Phase C: random starts


def _look_at(pos, target):
    """MuJoCo camera xyaxes for a camera at pos looking at target, z up."""
    fwd = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(fwd, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.concatenate([x, np.cross(x, fwd)]).tolist()


OBSTACLE_PARK = np.array([0.0, -1.5, -1.0])  # where the obstacle bodies sit while no obstacle is placed
EXTRA_FRACTIONS = (0.3, 0.7)  # line positions (start -> goal) of obstacles 1, 2; obstacle 0 is at the midpoint


def obstacle_name(i):
    return "obstacle" if i == 0 else f"obstacle{i}"


def build_model(orientation_goal=False, obstacle=False, wide_camera=False, n_obstacles=1) -> mujoco.MjModel:
    """Menagerie UR5e with the 2F-85 on its flange, a floor, a camera and a visual-only goal sphere.

    orientation_goal adds a visual-only pointer on the goal showing the target gripper axis; obstacle adds
    n_obstacles collidable spheres, each on its own mocap body (radii are set per episode)."""
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
    if wide_camera:
        wb.add_camera(name="wide", pos=WIDE_POS, xyaxes=_look_at(WIDE_POS, WIDE_TARGET))
    goal = wb.add_body(name="goal", mocap=True, pos=[0.0, 0.45, 0.25])
    goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=[0.1, 0.8, 0.2, 0.45],
                  contype=0, conaffinity=0)
    if orientation_goal:  # target gripper z-axis (approach direction), drawn from the goal up the gripper
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0.05, 0], pos=[0, 0, -0.05],
                      rgba=[0.1, 0.8, 0.2, 0.6], contype=0, conaffinity=0)
    for i in range(n_obstacles if obstacle else 0):
        obs_body = wb.add_body(name=obstacle_name(i), mocap=True, pos=OBSTACLE_PARK)
        obs_body.add_geom(name=obstacle_name(i), type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0],
                          rgba=[0.85, 0.2, 0.15, 0.7], contype=1, conaffinity=1)
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
                 width=480, height=360, random_start=False, observe_joints=False, orientation_goal=False,
                 angle_threshold_deg=15.0, goal_max_tilt_deg=45.0, rot_scale=0.1, obstacle=False,
                 obstacle_radius=(0.05, 0.08), obstacle_offset=0.03, collision_penalty=20.0, n_obstacles=1,
                 obstacle_spread=0.15):
        self.render_mode = render_mode
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.control_type = control_type
        self.n_substeps = n_substeps
        c, h = np.asarray(goal_center, dtype=np.float64), goal_range / 2
        self.goal_low, self.goal_high = c - h, c + h
        self.random_start, self.observe_joints = random_start, observe_joints
        self.orientation_goal, self.obstacle = orientation_goal, obstacle
        self.angle_threshold = np.deg2rad(angle_threshold_deg)
        self.goal_max_tilt, self.rot_scale = np.deg2rad(goal_max_tilt_deg), rot_scale
        self.obstacle_radius, self.obstacle_offset = obstacle_radius, obstacle_offset
        assert 1 <= n_obstacles <= 1 + len(EXTRA_FRACTIONS)
        self.n_obstacles, self.obstacle_spread = (n_obstacles if obstacle else 0), obstacle_spread
        self.collision_penalty = collision_penalty

        self.camera = "wide" if random_start else "side"
        self.model = build_model(orientation_goal, obstacle, wide_camera=random_start, n_obstacles=n_obstacles)
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
        self.robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] > 0
                                     and m.geom_contype[g] | m.geom_conaffinity[g]
                                     and m.body(m.geom_bodyid[g]).name != "goal"
                                     and not m.body(m.geom_bodyid[g]).name.startswith("obstacle")])
        if obstacle:
            names = [obstacle_name(i) for i in range(self.n_obstacles)]
            self.obstacle_geoms = np.array([m.geom(n).id for n in names])
            self.obstacle_mocaps = [m.body_mocapid[m.body(n).id] for n in names]
            self.obstacle_geom = self.obstacle_geoms[0]
            self.obstacles_pos = [OBSTACLE_PARK.copy() for _ in names]  # centre and radius of each sphere
            self.obstacles_r = [0.0] * self.n_obstacles

        n_act = 6 if control_type != "ee" or orientation_goal else 3
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        box = lambda n: spaces.Box(-10.0, 10.0, shape=(n,), dtype=np.float32)
        n_goal = 9 if orientation_goal else 3
        n_obs = 6 + 18 * observe_joints + 4 * self.n_obstacles
        self.observation_space = spaces.Dict(
            {"observation": box(n_obs), "achieved_goal": box(n_goal), "desired_goal": box(n_goal)})
        self.goal = np.zeros(n_goal)
        self.goal_quat = self.tcp_quat.copy()
        self._renderer = None
        self.width, self.height = width, height

    @property
    def obstacle_pos(self):  # sphere 0
        return self.obstacles_pos[0]

    @property
    def obstacle_r(self):
        return self.obstacles_r[0]

    # ---------------------------------------------------------------- kinematics
    def _set_qpos(self, d, q):
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def ik(self, target_pos, q0, iters=20, damping=0.05, tol=1e-4, target_quat=None, orient=True):
        """Damped least squares on the 6-D TCP error (position + orientation: target_quat, default gripper-down).
        orient=False solves for position only."""
        target_quat = self.tcp_quat if target_quat is None else target_quat
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
            mujoco.mju_mulQuat(dq_, target_quat, neg)
            mujoco.mju_quat2Vel(err[3:], dq_, 1.0)
            e = err if orient else err[:3]
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = (np.vstack([jacp, jacr]) if orient else jacp)[:, self.dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
            q = np.clip(q + dq, self.qlim[:, 0], self.qlim[:, 1])
        return q

    # ---------------------------------------------------------------- gym API
    def tcp_pos(self):
        return self.data.site_xpos[self.tcp].copy()

    def _tcp_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tcp, v, 0)
        return v[3:]  # world-frame linear velocity

    def tcp_rot6d(self, d=None):
        """First two columns of the TCP rotation matrix (the 6D orientation encoding)."""
        R = (self.data if d is None else d).site_xmat[self.tcp].reshape(3, 3)
        return np.concatenate([R[:, 0], R[:, 1]])

    def tcp_quat_now(self):
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.tcp])
        return q

    def achieved_goal(self):
        p = self.tcp_pos()
        return np.concatenate([p, self.tcp_rot6d()]) if self.orientation_goal else p

    def _obs(self):
        p = self.tcp_pos()
        parts = [p, self._tcp_vel()]
        if self.observe_joints:
            q = self.data.qpos[self.qadr]
            parts += [np.sin(q), np.cos(q), q / self.qlim[:, 1]]  # limits are symmetric
        for c, r in zip(self.obstacles_pos, self.obstacles_r) if self.obstacle else ():
            parts += [c, [r]]
        return {"observation": np.concatenate(parts).astype(np.float32),
                "achieved_goal": self.achieved_goal().astype(np.float32),
                "desired_goal": self.goal.astype(np.float32)}

    # ---------------------------------------------------------------- Phase C sampling
    def _robot_contacts(self, d, exclude=()):
        """Number of contacts involving a robot geom (self-contact or floor), ignoring geoms in exclude."""
        if d.ncon == 0:
            return 0
        g = d.contact.geom[:d.ncon]
        robot = np.isin(g, self.robot_geoms)
        other = np.isin(g, list(exclude)) if exclude else np.zeros_like(robot)
        return int((robot.any(axis=1) & ~other.any(axis=1)).sum())

    def _touches_obstacle(self, d):
        if d.ncon == 0:
            return False
        g = d.contact.geom[:d.ncon]
        hit = np.isin(g, self.obstacle_geoms).any(axis=1) & np.isin(g, self.robot_geoms).any(axis=1)
        return bool(hit.any())

    def sample_start(self, max_tries=10_000):
        """Rejection-sample arm joints within limits: TCP >= 5 cm above the floor, no robot contact."""
        d, m = self._ik_data, self.model
        exclude = tuple(self.obstacle_geoms) if self.obstacle else ()
        for _ in range(max_tries):
            q = self.np_random.uniform(self.qlim[:, 0], self.qlim[:, 1])
            d.qpos[:] = 0
            d.qpos[self.qadr] = q
            mujoco.mj_forward(m, d)
            if d.site_xpos[self.tcp][2] >= 0.05 and self._robot_contacts(d, exclude) == 0:
                return q
        raise RuntimeError("no valid start configuration found")

    def _sample_goal_quat(self):
        """Gripper-down rotated about a random axis by up to goal_max_tilt (geodesic distance from down)."""
        axis = self.np_random.normal(size=3)
        axis /= np.linalg.norm(axis)
        r, q = np.empty(4), np.empty(4)
        mujoco.mju_axisAngle2Quat(r, axis, self.np_random.uniform(0, self.goal_max_tilt))
        mujoco.mju_mulQuat(q, r, self.tcp_quat)  # world-frame rotation applied to gripper-down
        return q

    def _place_obstacle(self, start, goal_pos, tries=20):
        """Sphere 0 near the start -> goal midpoint, clear of the start TCP, the goal and the start arm pose;
        then the others near their line points (see the module docstring)."""
        lo, hi = self.obstacle_radius
        for _ in range(tries):
            r = self.np_random.uniform(lo, hi)
            c = (start + goal_pos) / 2 + self.np_random.uniform(-self.obstacle_offset, self.obstacle_offset, 3)
            if min(np.linalg.norm(c - start), np.linalg.norm(c - goal_pos)) < r + 0.06:
                continue
            self._set_obstacle(c, r)
            mujoco.mj_forward(self.model, self.data)
            if not self._touches_obstacle(self.data):
                break
        else:
            self._set_obstacle(OBSTACLE_PARK, 0.0)
            return False
        for i, f in zip(range(1, self.n_obstacles), EXTRA_FRACTIONS):
            if not self._place_extra(i, start + f * (goal_pos - start), start, goal_pos):
                for j in range(self.n_obstacles):
                    self._set_obstacle(OBSTACLE_PARK, 0.0, j)
                return False
        return True

    def _place_extra(self, i, p, start, goal_pos, tries=50):
        """Sphere i near line point p, its offset widening from obstacle_offset to obstacle_spread."""
        lo, hi = self.obstacle_radius
        for k in range(tries):
            off = self.obstacle_offset + (self.obstacle_spread - self.obstacle_offset) * k / (tries - 1)
            r = self.np_random.uniform(lo, hi)
            c = p + self.np_random.uniform(-off, off, 3)
            if min(np.linalg.norm(c - start), np.linalg.norm(c - goal_pos)) < r + 0.06:
                continue
            if any(np.linalg.norm(c - self.obstacles_pos[j]) <= r + self.obstacles_r[j] for j in range(i)):
                continue
            self._set_obstacle(c, r, i)
            mujoco.mj_forward(self.model, self.data)
            if not self._touches_obstacle(self.data):
                return True
        self._set_obstacle(OBSTACLE_PARK, 0.0, i)
        return False

    def _set_obstacle(self, c, r, i=0):
        m, g = self.model, self.obstacle_geoms[i]
        m.geom_size[g, 0] = m.geom_rbound[g] = max(r, 1e-3)
        m.geom_aabb[g, 3:] = max(r, 1e-3)
        self.data.mocap_pos[self.obstacle_mocaps[i]] = c
        self.obstacles_pos[i], self.obstacles_r[i] = np.array(c, dtype=np.float64), float(r)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.n_obstacles):
            self._set_obstacle(OBSTACLE_PARK, 0.0, i)
        while True:
            q0 = self.sample_start() if self.random_start else READY_QPOS
            self.data.qpos[self.qadr] = q0
            self.data.ctrl[self.arm_act] = q0
            goal_pos = self.np_random.uniform(self.goal_low, self.goal_high)
            if self.orientation_goal:
                self.goal_quat = self._sample_goal_quat()
                R = np.empty(9)
                mujoco.mju_quat2Mat(R, self.goal_quat)
                R = R.reshape(3, 3)
                self.goal = np.concatenate([goal_pos, R[:, 0], R[:, 1]])
                self.data.mocap_quat[self.goal_mocap] = self.goal_quat
            else:
                self.goal = goal_pos
            self.data.mocap_pos[self.goal_mocap] = goal_pos
            mujoco.mj_forward(self.model, self.data)
            if not self.obstacle or self._place_obstacle(self.tcp_pos(), goal_pos):
                break  # otherwise start and goal too close for an obstacle between them: resample both
        mujoco.mj_forward(self.model, self.data)
        obs = self._obs()
        info = {"is_success": self._is_success(obs["achieved_goal"], self.goal)}
        if self.obstacle:
            info["collision"] = False
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q = self.data.qpos[self.qadr].copy()
        if self.control_type == "ee":
            target = self.tcp_pos() + action[:3] * self.action_scale
            target[2] = max(target[2], 0.02)  # keep the fingertips off the floor (panda-gym clips z >= 0)
            target_quat = None
            if self.orientation_goal:  # rotate the current orientation by the commanded world-frame rotation
                target_quat, dq = np.empty(4), np.empty(4)
                mujoco.mju_axisAngle2Quat(dq, action[3:] / (np.linalg.norm(action[3:]) + 1e-12),
                                          np.linalg.norm(action[3:]) * self.rot_scale)
                mujoco.mju_mulQuat(target_quat, dq, self.tcp_quat_now())
            q_target = self.ik(target, q, target_quat=target_quat)
        else:
            q_target = np.clip(q + action * 0.05, self.qlim[:, 0], self.qlim[:, 1])
        self.data.ctrl[self.arm_act] = q_target
        collision = False
        if self.obstacle:  # check every physics step so a link cannot pass through the sphere unseen
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
                if self._touches_obstacle(self.data):
                    collision = True
                    break
        else:
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        obs = self._obs()
        info = {"collision": collision} if self.obstacle else {}
        success = self._is_success(obs["achieved_goal"], self.goal, info)
        info["is_success"] = success
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        return obs, reward, success or collision, False, info

    def _is_success(self, achieved, desired, info=None):
        return bool(self._goal_reached(achieved, desired) and not (info or {}).get("collision", False))

    def _goal_reached(self, achieved, desired):
        achieved, desired = np.asarray(achieved), np.asarray(desired)
        ok = np.linalg.norm(achieved[..., :3] - desired[..., :3], axis=-1) < self.distance_threshold
        if self.orientation_goal:
            ok &= rotation_angle(achieved[..., 3:], desired[..., 3:]) < self.angle_threshold
        return ok

    def compute_reward(self, achieved_goal, desired_goal, info):
        if not self.orientation_goal and not self.obstacle:  # Phase B
            d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
            if self.reward_type == "sparse":
                return -np.array(d > self.distance_threshold, dtype=np.float32)
            return -d.astype(np.float32)
        if self.reward_type != "sparse":
            raise NotImplementedError("Phase C options use the sparse reward only")
        r = -np.array(~self._goal_reached(achieved_goal, desired_goal), dtype=np.float32)
        if self.obstacle:
            r = np.where(collision_flags(info, r.shape), np.float32(-self.collision_penalty), r).astype(np.float32)
        return r

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

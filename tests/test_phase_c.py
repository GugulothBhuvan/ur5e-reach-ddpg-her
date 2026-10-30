"""Phase C env features: random start, joint observation, orientation goals, obstacle + HER penalty."""
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401
from envs.ur5e_reach import UR5eReachEnv

C2 = dict(random_start=True, observe_joints=True, goal_range=0.5, goal_center=(0.0, 0.45, 0.30))


def test_random_start_valid():
    e = UR5eReachEnv(**C2)
    starts = []
    for i in range(200):
        e.reset(seed=i)
        q = e.data.qpos[e.qadr].copy()
        starts.append(q)
        assert np.all(q >= e.qlim[:, 0]) and np.all(q <= e.qlim[:, 1])
        assert e.tcp_pos()[2] >= 0.05
        assert e._robot_contacts(e.data) == 0  # no self-contact, nothing touching the floor
        assert np.array_equal(e.data.ctrl[e.arm_act], q)  # servos hold the start pose
    starts = np.array(starts)
    assert starts.std(axis=0).min() > 0.5  # genuinely spread over joint space
    e.reset(seed=7)
    q7 = e.data.qpos[e.qadr].copy()
    e.reset(seed=7)
    assert np.array_equal(q7, e.data.qpos[e.qadr])  # reproducible from the seed


def test_joint_observation():
    e = UR5eReachEnv(**C2)
    obs, _ = e.reset(seed=0)
    assert obs["observation"].shape == (24,)
    q = e.data.qpos[e.qadr]
    j = obs["observation"][6:]
    assert np.allclose(j[:6], np.sin(q), atol=1e-6) and np.allclose(j[6:12], np.cos(q), atol=1e-6)
    assert np.allclose(j[12:], q / e.qlim[:, 1], atol=1e-6) and np.all(np.abs(j[12:]) <= 1)


@pytest.mark.parametrize("control_type", ["ee", "joints"])
def test_check_env_c2(control_type):
    check_env(UR5eReachEnv(control_type=control_type, **C2), skip_render_check=True)


def test_defaults_are_phase_b():
    e = UR5eReachEnv()
    assert e.observation_space["observation"].shape == (6,) and e.action_space.shape == (3,)
    assert e.observation_space["desired_goal"].shape == (3,)


# ---------------------------------------------------------------- C3: orientation goal
C3 = dict(C2, orientation_goal=True)


def _quat_of(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).ravel())
    return q


def test_rotation_angle_matches_quaternion_angle():
    from envs.ur5e_reach import rotation_angle
    rng = np.random.default_rng(0)
    a6, b6, ref = [], [], []
    for _ in range(64):
        qa, qb = rng.normal(size=4), rng.normal(size=4)
        qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
        Ra, Rb = np.empty(9), np.empty(9)
        mujoco.mju_quat2Mat(Ra, qa)
        mujoco.mju_quat2Mat(Rb, qb)
        Ra, Rb = Ra.reshape(3, 3), Rb.reshape(3, 3)
        a6.append(np.concatenate([Ra[:, 0], Ra[:, 1]]))
        b6.append(np.concatenate([Rb[:, 0], Rb[:, 1]]))
        ref.append(2 * np.arccos(min(1.0, abs(np.dot(qa, qb)))))
    assert np.allclose(rotation_angle(np.array(a6), np.array(b6)), ref, atol=1e-6)  # batched
    assert np.isclose(rotation_angle(a6[0], b6[0]), ref[0], atol=1e-6)  # single


def test_orientation_goal_spaces_and_targets():
    e = UR5eReachEnv(**C3)
    check_env(e, skip_render_check=True)
    assert e.action_space.shape == (6,) and e.observation_space["desired_goal"].shape == (9,)
    from envs.ur5e_reach import READY_QPOS, rotation_angle
    e._set_qpos(e._ik_data, READY_QPOS)
    down = e.tcp_rot6d(e._ik_data)
    tilts = []
    for i in range(200):
        obs, _ = e.reset(seed=i)
        tilts.append(rotation_angle(obs["desired_goal"][3:], down))
        assert np.allclose(obs["achieved_goal"][3:], e.tcp_rot6d(), atol=1e-6)
    assert max(tilts) <= np.deg2rad(45) + 1e-5 and max(tilts) > np.deg2rad(35)


def test_rotation_action_turns_gripper():
    e = UR5eReachEnv(orientation_goal=True)  # ready pose, so the gripper starts pointing down
    e.reset(seed=0)
    from envs.ur5e_reach import rotation_angle
    r0 = e.tcp_rot6d()
    p0 = e.tcp_pos()
    e.step(np.array([0, 0, 0, 1.0, 0, 0]))  # 0.1 rad about world x
    for _ in range(3):
        e.step(np.zeros(6))  # let the servos settle
    assert 0.04 < rotation_angle(e.tcp_rot6d(), r0) < 0.11  # ~0.06 of the 0.1 rad lands in one step, as ~3 of 5 cm
    assert np.linalg.norm(e.tcp_pos() - p0) < 0.01


def test_orientation_reward_batched_and_relabelled_matches_step():
    """HER relabels (next achieved goal, new desired goal) pairs: compute_reward on the batch must equal what
    step() would have returned had that goal been the episode's goal (replayed with the same actions)."""
    e = UR5eReachEnv(**C3)
    rng = np.random.default_rng(3)
    obs, _ = e.reset(seed=11)
    acts, ags = [], []
    for _ in range(30):
        g = e.goal.copy()
        a = rng.uniform(-1, 1, 6)
        obs, r, *_ = e.step(a)
        assert r == e.compute_reward(obs["achieved_goal"], g, {})
        acts.append(a)
        ags.append(obs["achieved_goal"])
    ags = np.array(ags)
    # relabel every transition with a goal achieved later in the episode (HER "future"), plus perturbations
    # that are near the position / angle thresholds so both success and failure occur
    new = ags[np.minimum(np.arange(30) + rng.integers(0, 5, 30), 29)].astype(np.float64)
    new[::3, :3] += rng.normal(0, 0.03, (10, 3))
    batch = e.compute_reward(ags, new, [{}] * 30)
    assert batch.shape == (30,) and set(np.unique(batch)) <= {-1.0, 0.0} and len(np.unique(batch)) == 2
    for t in range(30):
        e.reset(seed=11)  # same start
        e.goal = new[t]
        e.goal_quat = _quat_of(np.stack([new[t][3:6], new[t][6:9], np.cross(new[t][3:6], new[t][6:9])], axis=1))
        for k in range(t + 1):
            obs, r, term, _, info = e.step(acts[k])
        assert np.allclose(obs["achieved_goal"], ags[t], atol=1e-5)  # deterministic replay
        assert r == batch[t] and term == info["is_success"] == (r == 0.0)


# ---------------------------------------------------------------- C4: obstacle
C4 = dict(C3, obstacle=True)


def _collided_episode(e, seed0=0):
    """Run the scripted straight-line rule until an episode ends in a collision; return its transitions."""
    from baseline import scripted_action
    for seed in range(seed0, seed0 + 50):
        obs, _ = e.reset(seed=seed)
        steps = []
        for _ in range(50):
            a = scripted_action(e, obs)
            nxt, r, term, trunc, info = e.step(a)
            steps.append((obs, nxt, a, r, term, info))
            obs = nxt
            if term:
                break
        if info["collision"]:
            return steps
    raise AssertionError("no collision found")


def test_obstacle_placement():
    e = UR5eReachEnv(**C4)
    assert e.observation_space["observation"].shape == (28,)
    for i in range(100):
        obs, info = e.reset(seed=i)
        c, r = e.obstacle_pos, e.obstacle_r
        assert 0.05 <= r <= 0.08 and not info["collision"]
        assert np.allclose(obs["observation"][-4:], np.r_[c, r], atol=1e-6)
        assert np.linalg.norm(c - e.tcp_pos()) > r + 0.06 and np.linalg.norm(c - e.goal[:3]) > r + 0.06
        mid = (e.tcp_pos() + e.goal[:3]) / 2
        assert np.abs(c - mid).max() <= 0.03 + 1e-9  # near the midpoint of the straight line
        assert not e._touches_obstacle(e.data)  # start pose clear of the sphere
        assert e.model.geom_size[e.obstacle_geom, 0] == r


def test_collision_detection():
    e = UR5eReachEnv(obstacle=True)  # ready pose, fixed start
    e.reset(seed=0)
    # sphere 8 cm ahead of the TCP along +x: moving +x hits it, moving -x does not
    e._set_obstacle(e.tcp_pos() + [0.13, 0, 0], 0.06)
    obs, r, term, trunc, info = e.step(np.array([-1.0, 0, 0]))
    assert not info["collision"] and r == -1.0 and not term
    e.reset(seed=0)
    e._set_obstacle(e.tcp_pos() + [0.13, 0, 0], 0.06)
    hit = None
    for t in range(5):
        obs, r, term, trunc, info = e.step(np.array([1.0, 0, 0]))
        if info["collision"]:
            hit = t
            break
    assert hit is not None and term and not info["is_success"] and r == -20.0
    # success with a collision on the same step does not count: reward stays -20
    assert e.compute_reward(obs["desired_goal"], obs["desired_goal"], {"collision": True}) == -20.0
    assert e.compute_reward(obs["desired_goal"], obs["desired_goal"], {"collision": False}) == 0.0


def test_scripted_rule_hits_obstacle():
    e = UR5eReachEnv(**C4)
    steps = _collided_episode(e)
    obs, nxt, a, r, term, info = steps[-1]
    assert r == -20.0 and term and not info["is_success"]


@pytest.mark.parametrize("copy_info_dict", [True, False])
def test_her_keeps_collision_penalty(copy_info_dict):
    """A relabelled transition that collided still gets -20 when the collision flag reaches compute_reward
    (copy_info_dict=True). Without it (negative control) HER silently turns the crash into a success."""
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv
    e = UR5eReachEnv(**C4)
    venv = DummyVecEnv([lambda: e])
    buf = HerReplayBuffer(1000, venv.observation_space, venv.action_space, env=venv, n_sampled_goal=4,
                          goal_selection_strategy="future", copy_info_dict=copy_info_dict)
    steps = _collided_episode(e)
    batch = lambda o: {k: v[None] for k, v in o.items()}
    for obs, nxt, a, r, term, info in steps:
        buf.add(batch(obs), batch(nxt), a[None], np.array([r]), np.array([term]), [info])
    last = len(steps) - 1  # the colliding transition; "future" relabels it with its own achieved goal
    s = buf._get_virtual_samples(np.full(32, last), np.zeros(32, dtype=int))
    rewards = s.rewards.numpy().ravel()
    if copy_info_dict:
        assert np.all(rewards == -20.0)
    else:
        assert np.all(rewards == 0.0)  # the pitfall the flag avoids
    # non-colliding transitions are relabelled normally (-1 or 0, never -20)
    s = buf._get_virtual_samples(np.arange(last), np.zeros(last, dtype=int))
    assert set(np.unique(s.rewards.numpy())) <= {-1.0, 0.0}


# ---------------------------------------------------------------- stretch: several obstacles
C4_MULTI = dict(C4, n_obstacles=3)


@pytest.mark.parametrize("random_start", [True, False])
def test_multi_obstacle_placement(random_start):
    e = UR5eReachEnv(**dict(C4_MULTI, random_start=random_start))
    assert e.observation_space["observation"].shape == (6 + 18 + 12,)
    assert [e.model.geom(g).name for g in e.obstacle_geoms] == ["obstacle", "obstacle1", "obstacle2"]
    for i in range(100):
        obs, info = e.reset(seed=i)
        start, goal = e.tcp_pos(), e.goal[:3]
        assert not info["collision"] and not e._touches_obstacle(e.data)
        assert np.abs(e.obstacle_pos - (start + goal) / 2).max() <= 0.03 + 1e-9  # sphere 0 as in C4
        for k, (c, r) in enumerate(zip(e.obstacles_pos, e.obstacles_r)):
            assert 0.05 <= r <= 0.08 and e.model.geom_size[e.obstacle_geoms[k], 0] == r
            assert np.allclose(obs["observation"][24 + 4 * k: 28 + 4 * k], np.r_[c, r], atol=1e-6)
            assert np.linalg.norm(c - start) > r + 0.06 and np.linalg.norm(c - goal) > r + 0.06
            for j in range(k):
                assert np.linalg.norm(c - e.obstacles_pos[j]) > r + e.obstacles_r[j]
        for c, f in zip(e.obstacles_pos[1:], (0.3, 0.7)):
            assert np.abs(c - (start + f * (goal - start))).max() <= e.obstacle_spread + 1e-9


@pytest.mark.parametrize("k", [1, 2])
def test_multi_obstacle_collision(k):
    e = UR5eReachEnv(obstacle=True, n_obstacles=3)  # ready pose, fixed start
    e.reset(seed=0)
    for j in range(3):  # only sphere k in the TCP's path, the others out of the way
        e._set_obstacle(e.tcp_pos() + [0.13, 0, 0] if j == k else [0.0, -1.0 - j, 0.5], 0.06, j)
    hit = None
    for t in range(5):
        obs, r, term, trunc, info = e.step(np.array([1.0, 0, 0]))
        if info["collision"]:
            hit = t
            break
    assert hit is not None and term and not info["is_success"] and r == -20.0
    g = e.data.contact.geom[:e.data.ncon]
    assert (g == e.obstacle_geoms[k]).any() and not np.isin(g, np.delete(e.obstacle_geoms, k)).any()

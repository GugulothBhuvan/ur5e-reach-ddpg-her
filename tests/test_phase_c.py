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

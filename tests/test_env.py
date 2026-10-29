import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401
from envs.ur5e_reach import READY_QPOS, UR5eReachEnv


@pytest.fixture
def env():
    e = UR5eReachEnv()
    yield e
    e.close()


def test_ready_pose(env):
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs["achieved_goal"], (env.goal_low + env.goal_high) / 2, atol=2e-3)
    env._set_qpos(env._ik_data, READY_QPOS)
    z_axis = env._ik_data.site_xmat[env.tcp].reshape(3, 3)[:, 2]
    assert np.allclose(z_axis, [0, 0, -1], atol=1e-3)  # gripper pointing down


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_ik_moves_in_commanded_direction(env, axis, sign):
    env.reset(seed=0)
    p0 = env.tcp_pos()
    a = np.zeros(3)
    a[axis] = sign
    env.step(a)
    dp = env.tcp_pos() - p0
    assert sign * dp[axis] > 0.02  # ~3 cm of the commanded 5 cm in one control step from rest
    assert np.linalg.norm(np.delete(dp, axis)) < 0.01  # little motion off-axis


def test_orientation_held(env):
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, 3))
    z_axis = env.data.site_xmat[env.tcp].reshape(3, 3)[:, 2]
    assert z_axis[2] < -0.99


def test_compute_reward_matches_step(env):
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(1)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 3))
        assert r == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        assert term == info["is_success"] == (r == 0.0)
        if term:
            obs, _ = env.reset()
    # vectorised, as HER calls it
    ag, dg = rng.uniform(0, 0.5, (64, 3)), rng.uniform(0, 0.5, (64, 3))
    batch = env.compute_reward(ag, dg, [{}] * 64)
    assert batch.shape == (64,)
    assert np.array_equal(batch, [env.compute_reward(a, d, {}) for a, d in zip(ag, dg)])


def test_check_env(env):
    check_env(env, skip_render_check=True)


def test_random_steps_no_nan(env):
    env.reset(seed=3)
    rng = np.random.default_rng(2)
    for _ in range(1000):
        obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 3))
        assert all(np.isfinite(v).all() for v in obs.values()) and np.isfinite(r)
        if term:
            env.reset()


def test_goals_reachable_by_ik_controller():
    e = UR5eReachEnv()
    reached = 0
    for i in range(500):
        obs, _ = e.reset(seed=10_000 + i)
        for _ in range(50):
            obs, _, term, _, _ = e.step(np.clip((obs["desired_goal"] - obs["achieved_goal"]) / 0.05, -1, 1))
            if term:
                reached += 1
                break
    assert reached / 500 >= 0.99


def test_joint_control():
    e = UR5eReachEnv(control_type="joints")
    check_env(e, skip_render_check=True)
    e.reset(seed=0)
    q0 = e.data.qpos[e.qadr].copy()
    a = np.zeros(6)
    a[1] = 1.0  # shoulder lift
    for _ in range(3):
        e.step(a)
    dq = e.data.qpos[e.qadr] - q0
    assert 0.05 < dq[1] <= 0.15 + 1e-3  # up to 0.05 rad per step
    assert np.abs(np.delete(dq, 1)).max() < 0.02

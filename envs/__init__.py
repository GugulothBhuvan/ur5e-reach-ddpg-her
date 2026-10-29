from gymnasium.envs.registration import register

register(id="UR5eReach-v0", entry_point="envs.ur5e_reach:UR5eReachEnv", max_episode_steps=50)

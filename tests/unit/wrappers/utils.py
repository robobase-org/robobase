import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBS_NAME_FLAT1 = "obs0"
OBS_NAME_FLAT2 = "obs1"
OBS_NAME_IMG1 = "obs2"
OBS_NAME_IMG2 = "obs3"
OBS_SIZE = 100
IMG_SHAPE = (3, 8, 8)
ACTION_SHAPE = (2,)
EE_ACTION_SHAPE = (8,)


class DummyEnv(gym.Env):
    def __init__(self, episode_len: int = 5):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                OBS_NAME_FLAT1: spaces.Box(-1, episode_len, (OBS_SIZE,)),
                OBS_NAME_FLAT2: spaces.Box(-1, episode_len, (OBS_SIZE,)),
                OBS_NAME_IMG1: spaces.Box(-1, episode_len, IMG_SHAPE),
                OBS_NAME_IMG2: spaces.Box(-1, episode_len, IMG_SHAPE),
            }
        )
        self.action_space = spaces.Box(-2, 2, ACTION_SHAPE)
        self._steps = 0
        self._episode_len = episode_len

    def step(self, action):
        self._steps += 1
        flat_obs = self._steps + np.zeros(shape=(OBS_SIZE,))
        img_obs = self._steps + np.zeros(shape=IMG_SHAPE, dtype=np.uint8)
        return (
            {
                OBS_NAME_FLAT1: flat_obs,
                OBS_NAME_FLAT2: flat_obs,
                OBS_NAME_IMG1: img_obs,
                OBS_NAME_IMG2: img_obs,
            },
            0 if self._steps < self._episode_len else 100,
            self._steps >= self._episode_len,
            False,
            {},
        )

    def reset(self, *args, **kwargs):
        self._steps = 0
        flat_obs = np.zeros(shape=(OBS_SIZE,))
        img_obs = np.zeros(shape=IMG_SHAPE, dtype=np.uint8)
        return {
            OBS_NAME_FLAT1: flat_obs,
            OBS_NAME_FLAT2: flat_obs,
            OBS_NAME_IMG1: img_obs,
            OBS_NAME_IMG2: img_obs,
        }, {}


class DummyEEEnv(gym.Env):
    def __init__(self, episode_len: int = 5):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                OBS_NAME_FLAT1: spaces.Box(-1, 1, (OBS_SIZE,)),
                OBS_NAME_FLAT2: spaces.Box(-1, 1, (OBS_SIZE,)),
                OBS_NAME_IMG1: spaces.Box(-1, 1, IMG_SHAPE),
                OBS_NAME_IMG2: spaces.Box(-1, 1, IMG_SHAPE),
            }
        )
        act_min = np.array([-0.1, -0.5, 0.8] + 4 * [0.0] + [0.0], dtype=np.float32)
        act_max = np.array([0.7, 0.5, 1.7] + 4 * [1.0] + [1.0], dtype=np.float32)
        self.action_space = spaces.Box(act_min, act_max, EE_ACTION_SHAPE)
        self._steps = 0
        self._episode_len = episode_len


class DummyRewardEnv(DummyEnv):
    def __init__(self, episode_len: int = 5, default_reward: float | None = None):
        super().__init__(episode_len)
        self._default_reward = default_reward

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        reward = np.random.randint(-5, 5)
        if self._default_reward:
            reward = self._default_reward
        return obs, reward, terminated, truncated, info

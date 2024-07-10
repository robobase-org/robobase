"""Append Demo info."""
import gymnasium as gym
import numpy as np


class AppendDemoInfo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Append a demo flag to the info dict."""

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def _modify_info(self, info):
        if "demo" not in info:
            if self.is_vector_env:
                info["demo"] = np.zeros((self.num_envs,))
            else:
                info["demo"] = 0
        return info

    def reset(self, *args, **kwargs):
        """See base."""
        obs, info = self.env.reset(*args, **kwargs)
        return obs, self._modify_info(info)

    def step(self, action):
        """See base."""
        *rest, info = self.env.step(action)
        return *rest, self._modify_info(info)

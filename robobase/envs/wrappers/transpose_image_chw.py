"""Wrapper for adding time aware observations to environment observation."""
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class TransposeImageCHW(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Turn images from HWC to CHW."""

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self._vision_ndim = 4 if self.is_vector_env else 3
        for k, v in self.observation_space.items():
            if len(v.shape) == self._vision_ndim:
                self.observation_space[k] = spaces.Box(
                    0,
                    255,
                    dtype=np.uint8,
                    shape=(*v.shape[:-3], 3, v.shape[-3], v.shape[-2]),
                )

    def observation(self, observation, final: bool = False):
        """Adds to the observation with the current time step.

        Args:
            observation: The observation to add the time step to.
            final: If is final obs

        Returns:
            The observation with the time step appended to
        """
        for k, v in observation.items():
            if len(v.shape) == (self._vision_ndim - int(final)):
                observation[k] = v.transpose(*np.arange(0, v.ndim - 3), -1, -3, -2)
        return observation

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information
                from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "final_observation" in info:
            for fidx in np.where(info["_final_observation"])[0]:
                info["final_observation"][fidx] = self.observation(
                    info["final_observation"][fidx], True
                )
        return self.observation(observation), reward, terminated, truncated, info

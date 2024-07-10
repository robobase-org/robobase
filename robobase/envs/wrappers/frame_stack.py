from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original
    observation is an array with shape [3], so if we stack 4 observations, the
    processed observation
    has shape [4, 3].

    Note:
        - After :meth:`reset` is called, the frame buffer will be filled with the
          initial observation. I.e. the observation returned by :meth:`reset` will
          consist of `num_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
        """
        gym.utils.RecordConstructorArgs.__init__(self, num_stack=num_stack)
        gym.ObservationWrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.num_stack = num_stack
        self.frames = {}
        new_obs_dict = {}
        for name in self.observation_space.keys():
            orig_space = env.observation_space[name]
            self._axis = axis = 0
            shape = (num_stack,) + orig_space.shape
            if self.is_vector_env:
                self._axis = axis = 1
                shape = orig_space.shape[:1] + (num_stack,) + orig_space.shape[1:]
            new_obs_dict[name] = spaces.Box(
                np.expand_dims(orig_space.low, axis).repeat(num_stack, axis),
                np.expand_dims(orig_space.high, axis).repeat(num_stack, axis),
                shape=shape,
                dtype=orig_space.dtype,
            )
            self.frames[name] = np.zeros_like(new_obs_dict[name].sample())
        self.observation_space = spaces.Dict(new_obs_dict)

    def _add_frame(self, observation):
        for name, value in observation.items():
            if self.is_vector_env:
                self.frames[name] = np.concatenate(
                    [self.frames[name][:, 1:], np.expand_dims(value, 1)], 1
                )
            else:
                self.frames[name] = np.concatenate([self.frames[name][1:], [value]], 0)

    def _add_frame_at_idx(self, observation, idx: int = None):
        for name, value in observation.items():
            self.frames[name][idx] = np.concatenate(
                [self.frames[name][idx, 1:], np.expand_dims(value, 0)], 0
            )

    def observation(self, observation):
        return deepcopy(self.frames)

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
                self._add_frame_at_idx(info["final_observation"][fidx], fidx)
                info["final_observation"][fidx] = {
                    k: v[fidx]
                    for k, v in self.observation(info["final_observation"]).items()
                }
                single_agent_obs = {k: v[fidx] for k, v in observation.items()}
                [
                    self._add_frame_at_idx(single_agent_obs, fidx)
                    for _ in range(self.num_stack)
                ]
        self._add_frame(observation)
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)
        [self._add_frame(obs) for _ in range(self.num_stack)]
        return self.observation(obs), info

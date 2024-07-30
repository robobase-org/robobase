"""Concatenates dictionary of observations that share same shape."""
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict


class ConcatDim(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Concatenates dictionary of observations that share same shape."""

    def __init__(
        self,
        env: gym.Env,
        shape_length: int,
        dim: int,
        new_name: str,
        norm_obs: bool = False,
        obs_stats: dict = None,
        keys_to_ignore: list[str] = None,
    ):
        """Init.

        Args:
            env: The environment to apply the wrapper
            shape_length: The ndim we are interested in, e.g. images=3, low_dim=1.
            dim: The oberservations with this ...
            new_name: The name of the new observation.
            norm_obs: Whether to normalize observations.
            obs_stats: The obs statistics for normalizing observations.
            keys_to_ignore: A list of keys to not include in this combined observation,
                regardless if they meet shape_len.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self._shape_length = shape_length + int(self.is_vector_env)
        self._dim = dim + int(self.is_vector_env)
        self._new_name = new_name
        self._keys_to_ignore = [] if keys_to_ignore is None else keys_to_ignore
        self._norm_obs = norm_obs
        self._obs_stats = obs_stats
        new_obs_dict = {}
        combined = []
        for k, v in self.observation_space.items():
            if len(v.shape) == self._shape_length and k not in self._keys_to_ignore:
                combined.append(v)
            else:
                new_obs_dict[k] = v
        new_min = np.concatenate(list(map(lambda s: s.low, combined)), self._dim)
        new_max = np.concatenate(list(map(lambda s: s.high, combined)), self._dim)
        new_obs_dict[new_name] = Box(new_min, new_max, dtype=np.float32)
        self.observation_space = Dict(new_obs_dict)

    def _transform_timestep(self, observation, final: bool = False):
        shape_len = self._shape_length - int(final)
        dim = self._dim - int(final)
        new_obs = {}
        combined = []
        for k, v in observation.items():
            # We allow normalizing observations in the ConcatDim wrapper
            # because all obs stats are stored with original key names and
            # ConcatDim will rename them to new keys. Doing it here would
            # safer and cleaner.
            if len(v.shape) == shape_len and k not in self._keys_to_ignore:
                if self._norm_obs and k in self._obs_stats:
                    v = (v - self._obs_stats["mean"][k]) / self._obs_stats["std"][k]
                combined.append(v)
            else:
                new_obs[k] = v
        new_obs[self._new_name] = np.concatenate(combined, dim)
        return new_obs

    def observation(self, observation):
        """Adds to the observation with the current time step.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to
        """
        return self._transform_timestep(observation)

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        observations, *rest, info = super().step(action)
        if "final_observation" in info:
            for fidx in np.where(info["_final_observation"])[0]:
                info["final_observation"][fidx] = self._transform_timestep(
                    info["final_observation"][fidx], final=True
                )
        return self.observation(observations), *rest, info

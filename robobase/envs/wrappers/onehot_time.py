"""Wrapper for adding time aware observations to environment observation."""
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict


class OnehotTime(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Augment the observation with the current time step in the episode.
    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TimeAwareObservation
        >>> env = gym.make("CartPole-v1")
        >>> env = OnehotTime(env, episode_length=2)
        >>> env.reset(seed=42)
        (array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ,
                 1.        ,  0.        ,  0.        ]), {})
        >>> _ = env.action_space.seed(42)
        >>> env.step(env.action_space.sample())[0]
        array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,
                0         ,  1.        ,  0.        ])
    """

    PADDING = 2

    def __init__(self, env: gym.Env, episode_length: int):
        """Init.

        Args:
            env: The environment to apply the wrapper
            episode_length: The environment episode length
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self._episode_length = episode_length
        self.is_vector_env = getattr(env, "is_vector_env", False)
        extra_dim = (self.num_envs,) if self.is_vector_env else ()
        if isinstance(self.observation_space, Box):
            low = np.append(
                self.observation_space.low,
                extra_dim + tuple([0.0] * (episode_length + OnehotTime.PADDING)),
            )
            high = np.append(
                self.observation_space.high,
                extra_dim + tuple([1.0] * (episode_length + OnehotTime.PADDING)),
            )
            self.observation_space = Box(low, high, dtype=np.float32)
        elif isinstance(self.observation_space, Dict):
            self.observation_space["time"] = Box(
                0, 1, extra_dim + (episode_length + OnehotTime.PADDING,), dtype=np.uint8
            )
        else:
            raise ValueError("Unsupported space.")
        self._eye = np.eye(self._episode_length + OnehotTime.PADDING).astype(np.uint8)
        self._reset_t()

    def _reset_t(self):
        self._t = np.array([0], dtype=int)
        if self.is_vector_env:
            self._t = np.zeros(
                (
                    1,
                    self.num_envs,
                ),
                dtype=int,
            )

    def _transform_timestep(self, observation, t: np.ndarray = None):
        t = self._t if t is None else t
        observation["time"] = self._eye[t][0]
        return observation

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
        self._t += 1
        observations, rewards, terminations, truncations, info = self.env.step(action)
        if "final_observation" in info:
            for fidx in np.where(info["_final_observation"])[0]:
                info["final_observation"][fidx] = self._transform_timestep(
                    info["final_observation"][fidx], self._t[:, fidx]
                )
                self._t[:, fidx] = 0
        observations = self._transform_timestep(observations)
        return (
            observations,
            rewards,
            np.logical_or(
                terminations, truncations
            ),  # Required to end the episode when truncated
            False,  # Set to False not to bootstrap
            info,
        )

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        self._reset_t()
        return super().reset(**kwargs)

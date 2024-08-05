"""Todo."""
from typing import Dict
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, WrapperActType


class RescaleFromTanh(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Takes action from tanh space (-1 to 1) and transforms to env action space."""

    MIN = -1.0
    MAX = 1.0

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        action_space = env.action_space
        minimum = np.full(action_space.shape, -1, dtype=action_space.dtype)
        maximum = np.full(action_space.shape, 1, dtype=action_space.dtype)
        self.orig_action_space = self.env.action_space
        self.action_space = spaces.Box(
            minimum, maximum, shape=action_space.shape, dtype=action_space.dtype
        )
        self.is_vector_env = getattr(env, "is_vector_env", False)

    @staticmethod
    def transform_from_tanh(action, action_space):
        orig_min, orig_max = action_space.low, action_space.high
        scale = (orig_max - orig_min) / (RescaleFromTanh.MAX - RescaleFromTanh.MIN)
        new_action = orig_min + scale * (action - RescaleFromTanh.MIN)
        return new_action.astype(action.dtype, copy=False)

    @staticmethod
    def transform_to_tanh(action, action_space):
        orig_min, orig_max = action_space.low, action_space.high
        scale = (orig_max - orig_min) / (RescaleFromTanh.MAX - RescaleFromTanh.MIN)
        new_action = np.clip(
            ((action - orig_min) / scale) + RescaleFromTanh.MIN,
            RescaleFromTanh.MIN,
            RescaleFromTanh.MAX,
        )
        return new_action.astype(np.float32, copy=False)

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions
        """
        return RescaleFromTanh.transform_from_tanh(action, self.orig_action_space)


class RescaleFromTanhEEPose(
    RescaleFromTanh, gym.ActionWrapper, gym.utils.RecordConstructorArgs
):
    """Takes action from tanh space (-1 to 1) and transforms to env action space."""

    MIN = -1.0
    MAX = 1.0

    @staticmethod
    def transform_from_tanh(action, action_space):
        orig_min, orig_max = action_space.low, action_space.high
        scale = (orig_max - orig_min) / (
            RescaleFromTanhEEPose.MAX - RescaleFromTanhEEPose.MIN
        )
        new_action = orig_min + scale * (action - RescaleFromTanhEEPose.MIN)
        if new_action.ndim == 2:
            # Handle VecEnv
            magnitudes = np.linalg.norm(new_action[:, 3:7], axis=1)
            magnitudes[magnitudes == 0] = 1
            magnitudes = magnitudes[:, np.newaxis].repeat(4, axis=1)
            new_action[:, 3:7] = new_action[:, 3:7] / magnitudes
        else:
            # Normalize quaternions to unit magnitude
            magnitude = np.linalg.norm(new_action[3:7])
            # Avoid division by 0
            magnitude = 1 if magnitude == 0 else magnitude
            new_action[3:7] = new_action[3:7] / magnitude
        return new_action.astype(action.dtype, copy=False)

    @staticmethod
    def transform_to_tanh(action, action_space):
        # NOTE: assuming quaternions are already in unit magnitude
        assert np.allclose(np.linalg.norm(action[3:7]), 1)
        orig_min, orig_max = action_space.low, action_space.high
        scale = (orig_max - orig_min) / (
            RescaleFromTanhEEPose.MAX - RescaleFromTanhEEPose.MIN
        )
        new_action = ((action - orig_min) / scale) + RescaleFromTanhEEPose.MIN
        return new_action.astype(np.float32, copy=False)

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions
        """
        return RescaleFromTanhEEPose.transform_from_tanh(action, self.orig_action_space)


class RescaleFromTanhWithStandardization(
    gym.ActionWrapper, gym.utils.RecordConstructorArgs
):
    """Takes action from tanh space (-1 to 1), destandardize the action,
    and transforms to env action space."""

    CLIP = 3.0
    MIN = -1.0
    MAX = 1.0

    def __init__(self, env: gym.Env, action_stats: Dict[str, np.ndarray]):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        action_space = env.action_space
        minimum = np.full(action_space.shape, -1, dtype=action_space.dtype)
        maximum = np.full(action_space.shape, 1, dtype=action_space.dtype)
        self.action_space = spaces.Box(
            minimum, maximum, shape=action_space.shape, dtype=action_space.dtype
        )
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.action_stats = action_stats

    @staticmethod
    def transform_from_tanh(action, action_stats):
        # Scale back to standardized normal distribution, [-CLIP, CLIP]
        _clip = RescaleFromTanhWithStandardization.CLIP
        tanh_min, tanh_max = (
            RescaleFromTanhWithStandardization.MIN,
            RescaleFromTanhWithStandardization.MAX,
        )
        clip_min, clip_max = -_clip, _clip
        scale = (clip_max - clip_min) / (tanh_max - tanh_min)
        new_action = clip_min + scale * (action - tanh_min)

        # Destandardize to original action space
        action_mean, action_std = action_stats["mean"], action_stats["std"]
        if new_action.ndim == 2:
            new_action = new_action * action_std[None] + action_mean[None]
        else:
            new_action = new_action * action_std + action_mean
        return new_action.astype(action.dtype, copy=False)

    @staticmethod
    def transform_to_tanh(action, action_stats):
        # Standardize
        action_mean, action_std = action_stats["mean"], action_stats["std"]
        if action.ndim == 2:
            new_action = (action - action_mean[None]) / action_std[None]
        else:
            new_action = (action - action_mean) / action_std

        # Scale to Tanh space
        _clip = RescaleFromTanhWithStandardization.CLIP
        tanh_min, tanh_max = (
            RescaleFromTanhWithStandardization.MIN,
            RescaleFromTanhWithStandardization.MAX,
        )
        clip_min, clip_max = -_clip, _clip
        scale = (clip_max - clip_min) / (tanh_max - tanh_min)
        new_action = np.clip(
            ((new_action - clip_min) / scale) + tanh_min, tanh_min, tanh_max
        )
        return new_action.astype(np.float32, copy=False)

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions
        """
        return RescaleFromTanhWithStandardization.transform_from_tanh(
            action, self.action_stats
        )


class RescaleFromTanhWithMinMax(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Takes action from tanh space (-1 to 1), and transforms to env action space
    by reverting min/max normalization."""

    MIN = -1.0
    MAX = 1.0

    def __init__(
        self,
        env: gym.Env,
        action_stats: Dict[str, np.ndarray],
        min_max_margin: float = 0.0,
    ):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        action_space = env.action_space
        minimum = np.full(action_space.shape, -1, dtype=action_space.dtype)
        maximum = np.full(action_space.shape, 1, dtype=action_space.dtype)
        self.action_space = spaces.Box(
            minimum, maximum, shape=action_space.shape, dtype=action_space.dtype
        )
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.action_stats = action_stats
        self.min_max_margin = min_max_margin

    @staticmethod
    def transform_from_tanh(action, action_stats, min_max_margin):
        action = action.clip(-1.0, 1.0)
        action_min, action_max = action_stats["min"], action_stats["max"]
        _action_min = action_min - np.fabs(action_min) * min_max_margin
        _action_max = action_max + np.fabs(action_max) * min_max_margin

        new_action = (action + 1) / 2.0  # to [0, 1]
        new_action = new_action * (_action_max - _action_min) + _action_min  # original
        return new_action.astype(action.dtype, copy=False)

    @staticmethod
    def transform_to_tanh(action, action_stats, min_max_margin):
        action_min, action_max = action_stats["min"], action_stats["max"]
        _action_min = action_min - np.fabs(action_min) * min_max_margin
        _action_max = action_max + np.fabs(action_max) * min_max_margin

        new_action = (action - _action_min) / (
            _action_max - _action_min + 1e-8
        )  # to [0, 1]
        new_action = new_action * 2 - 1  # to [-1, 1]
        return new_action.astype(np.float32, copy=False)

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions
        """
        return RescaleFromTanhWithMinMax.transform_from_tanh(
            action, self.action_stats, self.min_max_margin
        )

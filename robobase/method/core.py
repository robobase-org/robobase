from abc import ABC, abstractmethod
from typing import Iterator, TypeAlias, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from robobase.intrinsic_reward_module.core import IntrinsicRewardModule
from robobase.replay_buffer.replay_buffer import ReplayBuffer


BatchedActionSequence: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.float32]
]
Metrics: TypeAlias = dict[str, np.ndarray]


class Method(nn.Module, ABC):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: torch.device,
        num_train_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module: Optional[IntrinsicRewardModule] = None,
        is_rl: bool = False,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.num_train_envs = num_train_envs
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.frame_stack_on_channel = frame_stack_on_channel
        self.num_eval_envs = 1  # Fixed to 1 for now
        self.intrinsic_reward_module = intrinsic_reward_module
        self._eval_env_running = False
        self.logging = False
        self.is_rl = is_rl

    @property
    def random_explore_action(self) -> torch.Tensor:
        # All actions live in -1 to 1, regardless of environment.
        min_action = -1
        max_action = 1
        return (min_action - max_action) * torch.rand(
            size=(self.num_train_envs,) + self.action_space.shape
        ) + max_action

    @abstractmethod
    def act(
        self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool
    ) -> BatchedActionSequence:
        pass

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> Metrics:
        pass

    @abstractmethod
    def reset(self, step: int, agents_to_reset: list[int]):
        pass

    @property
    def eval_env_running(self):
        return self._eval_env_running

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value


class ImitationLearningMethod(Method, ABC):
    pass


class OffPolicyMethod(Method, ABC):
    pass


class OnPolicyMethod(Method, ABC):
    """TODO: Leave open for future development."""

    pass


class ModelBasedMethod(Method, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

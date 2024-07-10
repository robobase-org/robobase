from __future__ import annotations

import numpy as np
from torch.utils.data import IterableDataset


class ReplayElement(object):
    def __init__(self, name, shape, type, is_observation=False):
        self.name = name
        self.shape = shape
        self.type = type
        self.is_observation = is_observation


class ReplayBuffer(IterableDataset):
    def replay_capacity(self):
        pass

    def batch_size(self):
        pass

    def get_storage_signature(self) -> tuple[list[ReplayElement], list[ReplayElement]]:
        pass

    def add(
        self,
        observation: dict,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncated: bool,
        **kwargs,
    ):
        """Adds a transition to the replay memory.

        Since the next_observation in the transition will be the observation added
        next, there is no need to pass it.
        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
          observation: current observation before action is applied.
          action: the action in the transition.
          reward: the reward received in the transition.
          terminal: Whether the transition was terminal or not.
          truncated: Whether the transition was truncated or not.
          kwargs: extra elements of the transition
        """
        pass

    def add_final(self, final_observation: dict):
        """Adds the final transition to the replay memory.

        Final transition only contains final observation, but no action, rewards and
        info as the episode has terminated.

        Args:
          final_observation: final observation of the episode
        """
        pass

    def is_empty(self):
        pass

    def is_full(self):
        pass

    def sample(self, batch_size=None, indices=None):
        """Sample transitions from replay buffer.

        Args:
            batch_size (int, optional): the batch size. Defaults to None.
            indices (list[int], optional): a list of global transition indices.
                                           Defaults to None.

        Returns:
            batch: a batch of transitions.
        """
        pass

    def shutdown(self):
        pass

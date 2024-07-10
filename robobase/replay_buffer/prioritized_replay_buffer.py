"""An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

from multiprocessing import Array, Value
from typing_extensions import override

import numpy as np

from robobase.replay_buffer.replay_buffer import ReplayElement
from robobase.replay_buffer.sum_tree import SumTree
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

PRIORITY = "priority"
SAMPLING_PROBABILITIES = "sampling_probabilities"


class PrioritizedReplayBuffer(UniformReplayBuffer):
    """An out-of-graph Replay Buffer for Prioritized Experience Replay.

    See uniform_replay_buffer.py for details.
    """

    def __init__(self, *args, **kwargs):
        """Initializes OutOfGraphPrioritizedReplayBuffer."""
        super(PrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self._sum_tree = SumTree(self._replay_capacity)
        self._num_to_sample = self.batch_size * (self._num_workers + 1)
        self._last_sampled_idx = Array("i", self._num_to_sample)
        self._times_samples = Value("i", 0)

    def get_storage_signature(
        self,
    ) -> tuple[dict[str, ReplayElement], dict[str, ReplayElement]]:
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          dict of ReplayElements defining the type of the contents stored.
        """
        storage_elements, obs_elements = super(
            PrioritizedReplayBuffer, self
        ).get_storage_signature()
        storage_elements[PRIORITY] = ReplayElement(PRIORITY, (), np.float32)
        return storage_elements, obs_elements

    @override
    def add(
        self,
        observation: dict,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncated: bool,
        priority: float = None,
        **kwargs,
    ):
        kwargs[PRIORITY] = priority
        if priority is None:
            priority = self._sum_tree.max_recorded_priority
            kwargs[PRIORITY] = priority
        self._sum_tree.set(self.add_count, priority)
        super(PrioritizedReplayBuffer, self).add(
            observation, action, reward, terminal, truncated, **kwargs
        )

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.

        For any memory location not yet used, the corresponding priority is 0.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).

        Returns:
          priorities: float, the corresponding priorities.
        """
        assert isinstance(indices, np.ndarray), "Indices must be an array."
        assert indices.shape, "Indices must be an array."
        assert indices.dtype == np.int32, "Indices must be int32s, " "given: {}".format(
            indices.dtype
        )
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self._sum_tree.get(memory_index)
        return priority_batch

    def sample_single(self, index=None):
        replay_sample = super().sample_single(index)
        if replay_sample is not None:
            replay_sample[SAMPLING_PROBABILITIES] = self._sum_tree.get(index)
        return replay_sample

    @override
    def sample(self, batch_size=None, indices=None):
        batch_size = self._batch_size if batch_size is None else batch_size
        if indices is not None and len(indices) != batch_size:
            raise ValueError(
                f"indices was of size {len(indices)}, but batch size was {batch_size}"
            )
        if indices is None:
            indices = self._sum_tree.stratified_sample(batch_size)
        samples = [self.sample_single(indices[i]) for i in range(batch_size)]
        batch = {}
        for k in samples[0].keys():
            batch[k] = np.stack([sample[k] for sample in samples])
        return batch

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
          priorities: float, the corresponding priorities.
        """
        assert (
            indices.dtype == np.int32
        ), "Indices must be integers, " "given: {}".format(indices.dtype)
        for index, priority in zip(indices, priorities):
            self._sum_tree.set(index, priority)

    def __iter__(self):
        while True:
            # Because not globally sampling, we can get repeat samples in a batch.
            # By sampling across the 10 highest priorities, this reduces that chance.
            sample = self.sample_single(
                self._sum_tree.stratified_sample(10)[np.random.randint(10)]
            )
            if sample is None:
                continue
            yield sample

"""Tests for prioritized_replay_buffer.py."""
import numpy as np
import pytest
from gymnasium import spaces
from torch.utils.data import DataLoader

from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer

# Default parameters used when creating the replay self._memory.
FRAME_STACKS = 4
ACTION_SEQ_LEN = 1
RGB_OBS_SHAPE = (FRAME_STACKS, 3, 84, 84)
RGB_OBS_DTYPE = np.uint8
STATE_OBS_SHAPE = (FRAME_STACKS, 9)
STATE_OBS_DTYPE = np.float32
ACTION_SHAPE = (ACTION_SEQ_LEN, 2)
BATCH_SIZE = 8


class TestPrioritizedReplayBuffer:
    def setup_method(self, method):
        self._test_single_obs_space = spaces.Dict(
            {"rgb": spaces.Box(0, 255, RGB_OBS_SHAPE, RGB_OBS_DTYPE)}
        )
        self._test_multi_obs_space = spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, RGB_OBS_SHAPE, RGB_OBS_DTYPE),
                "state": spaces.Box(-1, 1, STATE_OBS_SHAPE, STATE_OBS_DTYPE),
            }
        )
        # Slice of temporal axis
        self._test_single_obs = np.ones(RGB_OBS_SHAPE[1:], dtype=RGB_OBS_DTYPE) * 1
        self._test_action = np.ones(ACTION_SHAPE[1:], dtype=np.float32)
        self._test_reward = np.ones((), dtype=np.float32) * 2
        self._test_terminal = np.zeros((), dtype=np.int8)
        self._test_truncated = np.zeros((), dtype=np.int8)
        self._test_add_count = np.array(7)
        self._memory = None

    def teardown_method(self, method):
        if self._memory is not None:
            self._memory.shutdown()

    def test_add_with_and_without_priority(self):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )
        assert self._memory.add_count == 0
        priority = 1.0
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
            priority=priority,
        )
        assert self._memory.add_count == 1

    def test_add_with_additional_args_and_priority(self):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            extra_replay_elements=spaces.Dict(
                {
                    "extra1": spaces.Box(0, 5, (), np.uint8),
                    "extra2": spaces.Box(-1, 1, (2,), np.float32),
                }
            ),
            replay_capacity=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )
        priority = 1.0
        assert self._memory.add_count == 0
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
            priority=priority,
            extra1=np.array(1, dtype=np.uint8),
            extra2=np.array([-0.5, 0.5], dtype=np.float32),
        )
        assert self._memory.add_count == 1

    def test_get_priority_with_invalid_indices(self):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )
        assert self._memory.add_count == 0
        priority = 1.0
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
            priority=priority,
        )
        with pytest.raises(AssertionError) as e:
            self._memory.get_priority(0)
            assert "indices must be an array." in str(e.value)

    def test_set_and_get_priority(self):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=20,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )
        batch_size = 7
        indices = np.arange(0, batch_size).astype(np.int32)
        for index in range(batch_size):
            self._memory.add(
                {"rgb": self._test_single_obs},
                self._test_action,
                self._test_reward,
                self._test_terminal,
                self._test_truncated,
            )
        priorities = np.arange(batch_size)
        self._memory.set_priority(indices, priorities)
        # We send the indices in reverse order and verify the priorities come back
        # in that same order.
        fetched_priorities = self._memory.get_priority(np.flip(indices, 0))
        for i in range(batch_size):
            assert priorities[i] == fetched_priorities[batch_size - 1 - i]

    def test_new_element_has_high_priority(self):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=20,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
        )
        assert self._memory.get_priority(np.array([0], dtype=np.int32))[0] == 1.0

    def test_low_priority_element_not_frequently_sampled(self):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=20,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )
        # Add an item and set its priority to 0.
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
            priority=0.0,
        )
        # Now add a few new items.
        for _ in range(3):
            terminal = 1
            self._memory.add(
                {"rgb": self._test_single_obs},
                self._test_action,
                self._test_reward,
                terminal,
                self._test_truncated,
                priority=1.0,
            )
            self._memory.add_final({"rgb": self._test_single_obs})
        # This test should always pass.
        for _ in range(100):
            batch = self._memory.sample(batch_size=2)
            # Ensure all terminals are set to 1.
            assert (batch["terminal"] == 1).all()

    def _pytorch_dataloader_multi_worker(self, num_workers):
        self._memory = PrioritizedReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=200,
            nstep=1,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
        )
        episode_length = 10
        for i in range(episode_length):
            self._memory.add(
                {"rgb": self._test_single_obs * i},
                self._test_action,
                self._test_reward,
                self._test_terminal + float(i == (episode_length - 1)),
                self._test_truncated,
            )
        self._memory.add_final({"rgb": self._test_single_obs * (i + 1)})
        replay_loader = DataLoader(
            self._memory,
            batch_size=self._memory.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
        )
        assert self._memory.add_count == 10
        replay_iter = iter(replay_loader)
        batch = next(replay_iter)
        assert batch["rgb"].shape == (BATCH_SIZE,) + RGB_OBS_SHAPE

        num_episodes = 10
        for _ in range(num_episodes):
            for i in range(episode_length):
                self._memory.add(
                    {"rgb": self._test_single_obs * i},
                    self._test_action,
                    self._test_reward,
                    self._test_terminal + float(i == (episode_length - 1)),
                    self._test_truncated,
                )
            self._memory.add_final({"rgb": self._test_single_obs * (i + 1)})
        assert self._memory.add_count == 110
        assert "sampling_probabilities" in batch
        assert np.all(batch["sampling_probabilities"].numpy() == 1)
        self._memory.set_priority(
            batch["indices"].numpy().astype(np.int32), [10] * self._memory.batch_size
        )
        sampling_probabilities = []
        for _ in range(10):
            # We should eventually sample one of those high priority samples
            batch = next(replay_iter)
            sampling_probabilities.append(batch["sampling_probabilities"].numpy())
        assert np.any(np.reshape(sampling_probabilities, (-1,)) == 10)

    def test_pytorch_dataloader_multi_worker_1(self):
        self._pytorch_dataloader_multi_worker(1)

    def test_pytorch_dataloader_multi_worker_4(self):
        self._pytorch_dataloader_multi_worker(4)

"""Tests for uniform_replay_buffer.py."""

import copy
import numpy as np
import pytest
from gymnasium import spaces
from torch.utils.data import DataLoader

from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

# Default parameters used when creating the replay self._memory.
BATCH_LENGTH = 4
TRANSITION_SEQ_LEN = 4
ACTION_SEQ_LEN = 1
RGB_OBS_SHAPE = (1, 3, 84, 84)
RGB_OBS_DTYPE = np.uint8
STATE_OBS_SHAPE = (1, 9)
STATE_OBS_DTYPE = np.float32
ACTION_SHAPE = (ACTION_SEQ_LEN, 2)
BATCH_SIZE = 8


class TestSequentialReplayBuffer:
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
        self._memory1 = None
        self._memory2 = None
        self._memory3 = None

    def teardown_method(self, method):
        if self._memory is not None:
            self._memory.shutdown()

    def test_add(self):
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            sequential=True,
            transition_seq_len=TRANSITION_SEQ_LEN,
        )
        assert self._memory.add_count == 0
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
        )
        assert self._memory.add_count == 1

    def test_extra_add(self):
        self._memory = UniformReplayBuffer(
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
            sequential=True,
            transition_seq_len=TRANSITION_SEQ_LEN,
        )
        assert self._memory.add_count == 0
        self._memory.add(
            {"rgb": self._test_single_obs},
            self._test_action,
            self._test_reward,
            self._test_terminal,
            self._test_truncated,
            extra1=np.array(1, dtype=np.uint8),
            extra2=np.array([-0.5, 0.5], dtype=np.float32),
        )
        assert self._memory.add_count == 1
        with pytest.raises(ValueError) as e:
            self._memory.add(
                {"rgb": self._test_single_obs},
                self._test_action,
                self._test_reward,
                self._test_terminal,
                self._test_truncated,
                extra1=np.array(1, dtype=np.uint8),
            )
            assert "Add expects" in str(e.value)
        assert self._memory.add_count == 1

    def test_low_capacity(self):
        with pytest.raises(ValueError) as e:
            UniformReplayBuffer(
                observation_elements=self._test_single_obs_space,
                replay_capacity=3,
                action_shape=ACTION_SHAPE,
                batch_size=BATCH_SIZE,
                sequential=True,
                transition_seq_len=TRANSITION_SEQ_LEN,
            )
        assert "There is not enough capacity" in str(e.value)

        # We should be able to create a buffer that contains just enough for a
        # transition.
        UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            sequential=True,
            transition_seq_len=TRANSITION_SEQ_LEN,
        )

    def test_convert_episode_layout(self):
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=20,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            sequential=True,
            transition_seq_len=TRANSITION_SEQ_LEN,
        )

        episode = {
            "rgb": np.array([self._test_single_obs] * 5),
            "action": np.array([self._test_action] * 5),
            "reward": np.array([self._test_reward] * 5),
            "terminal": np.array([self._test_terminal] * 5),
            "truncated": np.array([self._test_truncated] * 5),
        }
        converted_episode = self._memory.convert_episode_layout(copy.deepcopy(episode))

        # Observation should not be modified
        assert np.allclose(converted_episode["rgb"], episode["rgb"])

        for key in ["action", "reward", "terminal", "truncated"]:
            # Check whether other values are shifted by one timestep
            assert np.allclose(converted_episode[key][1:], episode[key][:-1])

            # Check whether the initial values in converted episode are all zero
            assert np.allclose(
                converted_episode[key][:1], np.zeros_like(episode[key][:1])
            )

    def test_add_spillover_into_new_episode(self):
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=100,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            sequential=True,
            transition_seq_len=TRANSITION_SEQ_LEN,
        )
        # Episode length of 2 amd framestack of 4 means we need to spill over
        episode_length = 2
        num_episodes = 4
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
        batch = self._memory.sample_single()
        assert np.any(batch["is_first"] == 1)

    def _pytorch_dataloader_multi_worker(self, num_workers):
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=200,
            nstep=1,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            sequential=True,
            fetch_every=1,
            transition_seq_len=TRANSITION_SEQ_LEN,
        )

        episode_length = 5
        num_episodes = 2
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
        assert (
            batch["rgb"].shape == (BATCH_SIZE, TRANSITION_SEQ_LEN) + RGB_OBS_SHAPE[1:]
        )

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
        assert self._memory.add_count == 60
        _ = next(replay_iter)  # overcome preloading
        _ = next(replay_iter)  # overcome preloading
        _ = next(replay_iter)  # overcome preloading
        batch = next(replay_iter)
        assert np.any(batch["indices"].numpy() > 10)

    def test_pytorch_dataloader_multi_worker_1(self):
        self._pytorch_dataloader_multi_worker(1)

    def test_pytorch_dataloader_multi_worker_2(self):
        self._pytorch_dataloader_multi_worker(2)

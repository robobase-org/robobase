"""Tests for uniform_replay_buffer.py."""

import numpy as np
import pytest
from gymnasium import spaces
from torch.utils.data import DataLoader

from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

# Default parameters used when creating the replay self._memory.
FRAME_STACKS = 4
ACTION_SEQ_LEN = 1
RGB_OBS_SHAPE = (FRAME_STACKS, 3, 84, 84)
RGB_OBS_DTYPE = np.uint8
STATE_OBS_SHAPE = (FRAME_STACKS, 9)
STATE_OBS_DTYPE = np.float32
ACTION_SHAPE = (ACTION_SEQ_LEN, 2)
BATCH_SIZE = 8


class TestUniformReplayBuffer:
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
                nstep=1,
                action_shape=ACTION_SHAPE,
                batch_size=BATCH_SIZE,
            )
            assert "There is not enough capacity" in str(e.value)

        with pytest.raises(ValueError) as e:
            UniformReplayBuffer(
                observation_elements=self._test_single_obs_space,
                replay_capacity=5,
                nstep=5,
                action_shape=ACTION_SHAPE,
                batch_size=BATCH_SIZE,
            )
            assert "There is not enough capacity" in str(e.value)

        # We should be able to create a buffer that contains just enough for a
        # transition.
        UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=8,
            nstep=4,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
        )

    def test_nstep_reward_sum(self):
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=100,
            nstep=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            gamma=1.0,
        )
        episodes, episode_length = 4, 10
        for i in range(episodes):
            for j in range(episode_length):
                self._memory.add(
                    {"rgb": self._test_single_obs},
                    self._test_action,
                    self._test_reward,
                    self._test_terminal + float(j == (episode_length - 1)),
                    self._test_truncated,
                )
            self._memory.add_final({"rgb": self._test_single_obs})
        for i in range(100):
            batch = self._memory.sample()
            # Make sure the total reward is reward per step x update_horizon.
            assert batch["reward"][0] == 10.0

    def test_temporal_stack(self):
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=50,
            nstep=5,
            action_shape=ACTION_SHAPE,
            batch_size=BATCH_SIZE,
            gamma=1.0,
        )
        episode_length = 11
        for i in range(episode_length):
            self._memory.add(
                {"rgb": self._test_single_obs * i},
                self._test_action,
                self._test_reward,
                self._test_terminal + float(i == (episode_length - 1)),
                self._test_truncated,
            )
        self._memory.add_final({"rgb": self._test_single_obs * (i + 1)})
        batch = self._memory.sample(batch_size=1, indices=[1])
        assert batch["rgb"][0].shape == RGB_OBS_SHAPE
        # Check if the temporal axis is padded with the first observation
        assert np.all(batch["rgb"][0, 0] == batch["rgb"][0, 2])
        assert np.all(batch["rgb"][0, 1] == batch["rgb"][0, 2])
        assert np.all(batch["rgb"][0, 3] != batch["rgb"][0, 2])

    def test_pytorch_dataloader_single_worker(self):
        num_workers = 0
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=30,
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
        )
        assert self._memory.add_count == 10
        replay_iter = iter(replay_loader)
        batch = next(replay_iter)
        assert batch["rgb"].shape == (BATCH_SIZE,) + RGB_OBS_SHAPE
        for i in range(episode_length):
            self._memory.add(
                {"rgb": self._test_single_obs * i},
                self._test_action,
                self._test_reward,
                self._test_terminal + float(i == (episode_length - 1)),
                self._test_truncated,
            )
        self._memory.add_final({"rgb": self._test_single_obs * (i + 1)})
        assert self._memory.add_count == 20

    def test_pytorch_dataloader_multi_worker(self):
        num_workers = 1
        self._memory = UniformReplayBuffer(
            observation_elements=self._test_single_obs_space,
            replay_capacity=60,
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

        for i in range(episode_length):
            self._memory.add(
                {"rgb": self._test_single_obs * i},
                self._test_action,
                self._test_reward,
                self._test_terminal + float(i == (episode_length - 1)),
                self._test_truncated,
            )
        self._memory.add_final({"rgb": self._test_single_obs * (i + 1)})
        assert self._memory.add_count == 20

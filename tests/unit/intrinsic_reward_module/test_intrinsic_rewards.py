import pytest
import torch
from gymnasium import spaces

from robobase.intrinsic_reward_module.core import IntrinsicRewardModule
from robobase.intrinsic_reward_module import RND, ICM


OBS_LOW_DIM = "low_dim_state"
OBS_PIXELS_1 = "rgb_one"
OBS_PIXELS_2 = "rgb_two"
OBS_LOW_DIM_SIZE = 100
IMG_SHAPE = (3, 8, 8)
ACTION_SHAPE = (2, 4)
BATCH_SIZE = 10
TIME_SIZE = 2


@pytest.mark.parametrize(
    "intrinsic_module_cls,",
    [RND, ICM],
)
class TestIntrinsicRewards:
    def _sample_fake_batch(self, low_dim: bool = False, pixels: bool = False):
        batch = {
            "action": torch.rand(BATCH_SIZE, *ACTION_SHAPE, dtype=torch.float32),
        }
        if low_dim:
            batch.update(
                {
                    OBS_LOW_DIM: torch.rand(
                        BATCH_SIZE, TIME_SIZE, OBS_LOW_DIM_SIZE, dtype=torch.float32
                    ),
                    f"{OBS_LOW_DIM}_tp1": torch.rand(
                        BATCH_SIZE, TIME_SIZE, OBS_LOW_DIM_SIZE, dtype=torch.float32
                    ),
                }
            )
        if pixels:
            batch.update(
                {
                    OBS_PIXELS_1: torch.rand(
                        BATCH_SIZE, TIME_SIZE, *IMG_SHAPE, dtype=torch.float32
                    ),
                    f"{OBS_PIXELS_1}_tp1": torch.rand(
                        BATCH_SIZE, TIME_SIZE, *IMG_SHAPE, dtype=torch.float32
                    ),
                    OBS_PIXELS_2: torch.rand(
                        BATCH_SIZE, TIME_SIZE, *IMG_SHAPE, dtype=torch.float32
                    ),
                    f"{OBS_PIXELS_2}_tp1": torch.rand(
                        BATCH_SIZE, TIME_SIZE, *IMG_SHAPE, dtype=torch.float32
                    ),
                }
            )
        return batch

    def test_low_dim_obs(self, intrinsic_module_cls: type[IntrinsicRewardModule]):
        observation_space = spaces.Dict(
            {
                OBS_LOW_DIM: spaces.Box(-1, 1, (TIME_SIZE, OBS_LOW_DIM_SIZE)),
            }
        )
        action_space = spaces.Box(-2, 2, ACTION_SHAPE)
        rew = intrinsic_module_cls(
            observation_space=observation_space,
            action_space=action_space,
            device=torch.device("cpu"),
        )
        batch = self._sample_fake_batch(low_dim=True)
        rewards = rew.compute_irs(batch)
        rew.update(batch)
        assert rewards.shape == (BATCH_SIZE, 1)
        assert rewards.dtype == torch.float32

    def test_pixel_obs(self, intrinsic_module_cls: type[IntrinsicRewardModule]):
        observation_space = spaces.Dict(
            {
                OBS_PIXELS_1: spaces.Box(-1, 1, (TIME_SIZE, *IMG_SHAPE)),
                OBS_PIXELS_2: spaces.Box(-1, 1, (TIME_SIZE, *IMG_SHAPE)),
            }
        )
        action_space = spaces.Box(-2, 2, ACTION_SHAPE)
        rew = intrinsic_module_cls(
            observation_space=observation_space,
            action_space=action_space,
            device=torch.device("cpu"),
        )
        batch = self._sample_fake_batch(pixels=True)
        rewards = rew.compute_irs(batch)
        rew.update(batch)
        assert rewards.shape == (BATCH_SIZE, 1)
        assert rewards.dtype == torch.float32

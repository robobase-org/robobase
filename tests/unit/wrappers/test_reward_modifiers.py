import numpy as np
from tests.unit.wrappers.utils import DummyRewardEnv
from robobase.envs.wrappers import ClipReward, ScaleReward


def test_clip_env():
    env = ClipReward(DummyRewardEnv(), -1, 1)
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert -1 <= reward <= 1


def test_scale_env():
    default_reward = np.random.random()
    scale = np.random.random()
    env = ScaleReward(DummyRewardEnv(default_reward=default_reward), scale=scale)
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert np.isclose(reward, default_reward * scale)

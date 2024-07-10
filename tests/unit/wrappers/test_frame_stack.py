import numpy as np
from gymnasium.vector import SyncVectorEnv
from tests.unit.wrappers.utils import DummyEnv, OBS_SIZE, OBS_NAME_FLAT1
from robobase.envs.wrappers import FrameStack

NUM_STACK = 5
NUM_ENVS = 2


def test_onehot_single_env():
    env = FrameStack(DummyEnv(), NUM_STACK)
    assert env.observation_space[OBS_NAME_FLAT1].shape == (NUM_STACK, OBS_SIZE)
    obs, _ = env.reset()
    assert obs[OBS_NAME_FLAT1].shape == (NUM_STACK, OBS_SIZE)
    obs, *_ = env.step(env.action_space.sample())
    assert obs[OBS_NAME_FLAT1].shape == (NUM_STACK, OBS_SIZE)


def test_onehot_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: FrameStack(DummyEnv(), NUM_STACK) for _ in range(NUM_ENVS)]
    )
    assert env.observation_space[OBS_NAME_FLAT1].shape == (
        NUM_ENVS,
        NUM_STACK,
        OBS_SIZE,
    )
    obs, info = env.reset()
    assert obs[OBS_NAME_FLAT1].shape == (NUM_ENVS, NUM_STACK, OBS_SIZE)
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert obs[OBS_NAME_FLAT1].shape == (NUM_ENVS, NUM_STACK, OBS_SIZE)
    assert len(info["final_observation"]) == NUM_ENVS
    assert info["final_observation"][0][OBS_NAME_FLAT1].shape == (NUM_STACK, OBS_SIZE)


def test_onehot_wrapped_vec_env():
    env = FrameStack(
        SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]), NUM_STACK
    )
    assert env.observation_space[OBS_NAME_FLAT1].shape == (
        NUM_ENVS,
        NUM_STACK,
        OBS_SIZE,
    )
    obs, info = env.reset()
    assert obs[OBS_NAME_FLAT1].shape == (NUM_ENVS, NUM_STACK, OBS_SIZE)
    assert np.all(obs[OBS_NAME_FLAT1] == 0)
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert obs[OBS_NAME_FLAT1].shape == (NUM_ENVS, NUM_STACK, OBS_SIZE)
    assert len(info["final_observation"]) == NUM_ENVS
    assert np.all(
        info["final_observation"][0][OBS_NAME_FLAT1] == np.arange(1, 6)[:, np.newaxis]
    )
    assert info["final_observation"][0][OBS_NAME_FLAT1].shape == (NUM_STACK, OBS_SIZE)

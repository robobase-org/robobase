from gymnasium.vector import SyncVectorEnv
from tests.unit.wrappers.utils import DummyEnv
from robobase.envs.wrappers import OnehotTime

NUM_ENVS = 2
EPISODE_LEN = 5


def test_onehot_single_env():
    env = OnehotTime(DummyEnv(), EPISODE_LEN)
    assert env.observation_space["time"].shape == (EPISODE_LEN + OnehotTime.PADDING,)
    obs, _ = env.reset()
    assert obs["time"].shape == (EPISODE_LEN + OnehotTime.PADDING,)
    obs, *_ = env.step(env.action_space.sample())
    assert obs["time"].shape == (EPISODE_LEN + OnehotTime.PADDING,)


def test_onehot_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: OnehotTime(DummyEnv(), EPISODE_LEN) for _ in range(NUM_ENVS)]
    )
    assert env.observation_space["time"].shape == (
        NUM_ENVS,
        EPISODE_LEN + OnehotTime.PADDING,
    )
    obs, info = env.reset()
    assert obs["time"].shape == (NUM_ENVS, EPISODE_LEN + OnehotTime.PADDING)
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert obs["time"].shape == (NUM_ENVS, EPISODE_LEN + OnehotTime.PADDING)
    assert len(info["final_observation"]) == NUM_ENVS
    assert info["final_observation"][0]["time"].shape == (
        EPISODE_LEN + OnehotTime.PADDING,
    )


def test_onehot_wrapped_vec_env():
    env = OnehotTime(
        SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]), EPISODE_LEN
    )
    assert env.observation_space["time"].shape == (
        NUM_ENVS,
        EPISODE_LEN + OnehotTime.PADDING,
    )
    obs, info = env.reset()
    assert obs["time"].shape == (NUM_ENVS, EPISODE_LEN + OnehotTime.PADDING)
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert obs["time"].shape == (NUM_ENVS, EPISODE_LEN + OnehotTime.PADDING)
    assert len(info["final_observation"]) == NUM_ENVS
    assert info["final_observation"][0]["time"].shape == (
        EPISODE_LEN + OnehotTime.PADDING,
    )

import pytest
from gymnasium.vector import SyncVectorEnv
from tests.unit.wrappers.utils import DummyEnv, ACTION_SHAPE
from robobase.envs.wrappers import ActionSequence, RecedingHorizonControl

NUM_ENVS = 2
SEQ_LEN = 5
TIME_LIMIT = 100
EXE_LEN = 1


def _create_receding_horizon_env():
    env = RecedingHorizonControl(
        DummyEnv(),
        sequence_length=SEQ_LEN,
        time_limit=TIME_LIMIT,
        execution_length=EXE_LEN,
        temporal_ensemble=True,
    )
    return env


def test_action_sequence_has_correct_shape():
    env = ActionSequence(DummyEnv(), SEQ_LEN)
    assert env.action_space.shape == (SEQ_LEN,) + ACTION_SHAPE


def test_action_sequence_vec_has_correct_shape():
    env = SyncVectorEnv(
        [lambda: ActionSequence(DummyEnv(), SEQ_LEN) for _ in range(NUM_ENVS)]
    )
    assert env.action_space.shape == (NUM_ENVS, SEQ_LEN) + ACTION_SHAPE


def test_action_sequence_can_step():
    env = ActionSequence(DummyEnv(), SEQ_LEN)
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert "action_sequence_mask" in info
        assert info["action_sequence_mask"].shape == (SEQ_LEN,)


def test_action_sequence_can_step_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: ActionSequence(DummyEnv(), SEQ_LEN) for _ in range(NUM_ENVS)]
    )
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert "action_sequence_mask" in info["final_info"][0]
        assert info["final_info"][0]["action_sequence_mask"].shape == (SEQ_LEN,)


def test_action_sequence_cant_be_used_with_vec_env():
    with pytest.raises(NotImplementedError):
        ActionSequence(
            SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]), SEQ_LEN
        )


def test_receding_horizon_has_correct_shape():
    env = _create_receding_horizon_env()
    assert env.action_space.shape == (SEQ_LEN,) + ACTION_SHAPE


def test_receding_horizon_vec_has_correct_shape():
    env = SyncVectorEnv(
        [lambda: _create_receding_horizon_env() for _ in range(NUM_ENVS)]
    )
    assert env.action_space.shape == (NUM_ENVS, SEQ_LEN) + ACTION_SHAPE


def test_receding_horizon_can_step():
    env = _create_receding_horizon_env()
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        assert "action_sequence_mask" in info
        assert info["action_sequence_mask"].shape == (SEQ_LEN,)


def test_receding_horizon_can_step_vec_wrapped_env():
    env = SyncVectorEnv(
        [lambda: _create_receding_horizon_env() for _ in range(NUM_ENVS)]
    )
    env.reset()
    for _ in range(5):
        obs, *_, info = env.step(env.action_space.sample())
        if "final_info" in info:
            assert "action_sequence_mask" in info["final_info"][0]
            assert info["final_info"][0]["action_sequence_mask"].shape == (SEQ_LEN,)
        else:
            assert "action_sequence_mask" in info
            assert info["action_sequence_mask"][0].shape == (SEQ_LEN,)


def test_receding_horizon_cant_be_used_with_vec_env():
    with pytest.raises(NotImplementedError):
        RecedingHorizonControl(
            SyncVectorEnv([lambda: DummyEnv() for _ in range(NUM_ENVS)]),
            sequence_length=SEQ_LEN,
            time_limit=TIME_LIMIT,
            execution_length=EXE_LEN,
            temporal_ensemble=True,
        )

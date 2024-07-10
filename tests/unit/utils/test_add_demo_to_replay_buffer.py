import numpy as np
from gymnasium import spaces
import pytest

from robobase.utils import add_demo_to_replay_buffer
from robobase.envs.env import DemoEnv
from tests.unit.wrappers.utils import DummyEnv
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from robobase.envs.wrappers import FrameStack, RecedingHorizonControl, ActionSequence


def collect_demo_from_dummy_env(env, num_demo):
    traj = []

    for traj_idx in range(num_demo):
        obs, info = env.reset()
        info["demo"] = 1

        cur_traj = [[obs, info]]
        term = 0
        trunc = 0
        while not term and not trunc:
            # Divide by 100 to make it within [-1, 1] range
            action = np.ones_like(env.action_space.sample()) * (traj_idx - 2) / 100.0
            obs, rew, term, trunc, info = env.step(action)
            info["demo_action"] = action
            info["demo"] = 1
            cur_traj.append([obs, rew, term, trunc, info])

        traj.append(cur_traj)

    return traj


def wrap_env(env, frame_stack, action_sequence, execution_step, demo_env=False):
    if not demo_env:
        env = FrameStack(env, frame_stack)
        if action_sequence == execution_step:
            env = ActionSequence(env, action_sequence)
        else:
            env = RecedingHorizonControl(
                env, action_sequence, 5, execution_step, temporal_ensemble=False
            )

    return env


@pytest.mark.parametrize(
    "frame_stack, action_sequence, execution_step",
    [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (3, 3, 1), (2, 5, 1), (2, 5, 5)],
)
def test_add_demo_to_replay_buffer(frame_stack, action_sequence, execution_step):
    eps_len = 20
    num_demo = 5

    # Check demo collection
    env = DummyEnv(episode_len=eps_len)
    demos = collect_demo_from_dummy_env(env, num_demo=5)
    assert len(demos) == num_demo
    for demo in demos:
        assert (
            len(demo) == eps_len + 1
        )  # demo length should be 1 + ep_len due to first reset

    demo_env = DemoEnv(demos, env.action_space, env.observation_space)

    # For non-demo env, action_seq and frame_stack should be set properly
    env = wrap_env(env, frame_stack, action_sequence, execution_step, demo_env=False)
    print(env.action_space.shape)
    print(env.observation_space)
    assert env.action_space.shape[0] == action_sequence
    for obs_space in env.observation_space.values():
        assert obs_space.shape[0] == frame_stack

    # For demo env, action_seq and frame_stack should always be 1
    demo_env = wrap_env(
        demo_env, frame_stack, action_sequence, execution_step, demo_env=True
    )
    print(demo_env.action_space.shape)
    print(demo_env.observation_space)

    info_elements = spaces.Dict({})
    info_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
    replay_buffer = UniformReplayBuffer(
        action_shape=env.action_space.shape,
        action_dtype=env.action_space.dtype,
        nstep=1,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=env.observation_space,
        extra_replay_elements=info_elements,
    )
    # print(replay_buffer._frame_stacks, replay_buffer._action_seq_len)
    assert replay_buffer._frame_stacks == frame_stack
    assert replay_buffer._action_seq_len == action_sequence

    # Test adding demo to replay buffer
    for _ in range(len(demos)):
        add_demo_to_replay_buffer(demo_env, replay_buffer)
    assert replay_buffer.add_count == eps_len * num_demo
    assert replay_buffer._num_episodes == num_demo

    # Sample episode from replay buffer
    replay_buffer._try_fetch()
    episode, _ = replay_buffer._sample_episode()
    for v in episode.values():
        assert len(v) == eps_len + 1  # Episode length should be correct

    # Sample transitions from replay buffer
    for _ in range(10):
        sample = replay_buffer.sample_single()
        print(sample["action"])
        # - Frame stack, and action sequence should be correctly implemented
        for key in env.observation_space.keys():
            assert sample[key].shape == env.observation_space[key].shape
        assert sample["action"].shape == env.action_space.shape

    # Check action_sequence is properly padded with zero
    for idx in range(eps_len):
        sample = replay_buffer.sample_single(idx)
        print(sample["action"])
        num_pad_pos = max(idx - (eps_len - action_sequence), 0)
        if num_pad_pos > 0:
            assert sample["action"][-num_pad_pos:].sum() == 0


if __name__ == "__main__":
    test_add_demo_to_replay_buffer(2, 1, 1)
    test_add_demo_to_replay_buffer(2, 5, 1)
    test_add_demo_to_replay_buffer(2, 5, 5)

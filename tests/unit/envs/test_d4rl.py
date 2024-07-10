import pytest
import gym as gym_old
import gymnasium as gym
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

from robobase.envs.d4rl import (
    get_traj_dataset,
    D4RLEnvCompatibility,
    ConvertObsToDict,
    D4RLEnvFactory,
)


@pytest.mark.parametrize(
    "task_name, expected_len",
    [
        ("halfcheetah-medium-v2", 1000),
        ("hopper-medium-v2", 2187),
        ("walker2d-medium-v2", 1191),
    ],
)
def test_get_trajectory_dataset(task_name, expected_len):
    env = gym_old.make(task_name)
    d4rl_trajs, _ = get_traj_dataset(env)

    for traj in d4rl_trajs:  # for each trajecotry
        assert len(traj[0]) == 2  # first transition only contains obs and info
        for i in range(1, len(traj)):
            assert len(traj[i]) == 5  # subsequent transitons contain 5 items
            assert "demo_action" in traj[i][4]
            assert traj[i][4]["demo"] == 1

    assert len(d4rl_trajs) == expected_len


@pytest.mark.parametrize(
    "task_name",
    [("halfcheetah-medium-v2"), ("hopper-medium-v2"), ("walker2d-medium-v2")],
)
def test_env_compatilibility_wrapper(task_name):
    env = gym_old.make(task_name)
    env = D4RLEnvCompatibility(env)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32

    env.reset()
    action = env.action_space.sample()
    res = env.step(action)
    assert len(res) == 5  # In the new gym api, step should return 5 items.


@pytest.mark.parametrize(
    "task_name", [("HalfCheetah-v4"), ("Hopper-v4"), ("Walker2d-v4")]
)
def test_convert_obs_to_dict_wrapper(task_name):
    env = gym.make(task_name)
    env = ConvertObsToDict(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)

    obs, info = env.reset()
    assert "low_dim_state" in obs

    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    assert "low_dim_state" in obs


@pytest.fixture()
def compose_cfg():
    GlobalHydra.instance().clear()
    initialize(config_path="../../../robobase/cfgs")
    method = ["method=" + "iql_drqv2"]
    cfg = compose(
        config_name="robobase_config",
        overrides=method
        + [
            "pixels=false",
            "env=d4rl/hopper",
            "save_snapshot=true",
            "snapshot_every_n=1",
        ],
    )
    return cfg


@pytest.mark.parametrize(
    "num_demos, desired_num_demos", [(float("inf"), 2187), (0, 0), (100, 100)]
)
def test_collect_demo(num_demos, desired_num_demos, compose_cfg):
    factory = D4RLEnvFactory()
    factory.collect_or_fetch_demos(compose_cfg, num_demos)

    assert len(factory._raw_demos) == desired_num_demos

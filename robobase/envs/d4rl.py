from typing import Optional, List
import copy
import collections
import logging
import random
import math
import multiprocessing as mp

import numpy as np
import gym as gym_old
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
import d4rl
from omegaconf import DictConfig
from gymnasium.wrappers import TimeLimit, EnvCompatibility

from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    ActionSequence,
    AppendDemoInfo,
)
from robobase.utils import add_demo_to_replay_buffer
from robobase.envs.env import EnvFactory, DemoEnv

SUPPORTED_ENVS = ["ant", "antmaze", "halfcheetah", "hopper", "walker2d"]

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
        episode_return += rew

    return episode_return


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def get_traj_dataset(env, sorting=True):
    dataset = D4RLDataset(env)
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )
    if sorting:
        trajs.sort(key=compute_returns)

    # Convert traj to RoboBase demo
    converted_trajs = [[]]
    for traj in trajs:
        # The first transition only contains (obs, info),
        # corresponding to the ouput of env.reset()
        converted_trajs[-1].append([traj[0][0], {"demo": 1}])

        # For the subsequent transitions. we convert
        # (obs, actions, rew, masks, dones_float, next_obs)
        # to (next_obs, rew, term, trunc, next_info) required by robobase.DemoEnv.
        for ts in traj:
            # truncation is always False as the time limit is handled by
            # the `TimeLimit` wrapper.
            converted_trajs[-1].append(
                [ts[5], ts[2], ts[4], False, {"demo_action": ts[1], "demo": 1}]
            )

        # If traj length equals to max_episode_len, then termination=False and
        # truncated=True
        # NOTE: For d4rl, the collected trajectory has 1 less step then
        # max_episode_steps.
        if len(traj) == env.spec.max_episode_steps - 1:
            converted_trajs[-1][-1][2] = False

        converted_trajs.append([])
    converted_trajs.pop()  # Remove the last empty traj

    # NOTE: this raw_dataset is not sorted
    return converted_trajs, dataset.raw_dataset


class D4RLDataset:
    def __init__(self, env: gym_old.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        logging.warning("Collecting dataset from d4rl")
        self.raw_dataset = dataset = d4rl.qlearning_dataset(env.env)

        # Clip actions.
        # NOTE: sometimes action could be 1 which is not reachable for a tanh policy.
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        # Fix d4rl termination state
        # NOTE: Due to dataset bugs, we manually add termination flag if next
        # observation is far away from current.
        dones_float = np.zeros_like(dataset["rewards"])
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(np.float32)
        self.size = len(dataset["observations"])


class ConvertObsToDict(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    A wrapper to warp raw observation to space.Dict with key "low_dim_state".
    """

    def __init__(self, env: gym.Env):
        """Init.

        Args:
            env (gym.Env): the environment to apply wrapper on.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.env = env

        self.observation_space = spaces.Dict(
            {"low_dim_state": self.env.observation_space}
        )

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: the action to take

        Returns:
            The environment's step using the action, with observation wrapped in a
            Dict with key "low_dim_state".
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment,  with observation wrapped in a Dict with key
            "low_dim_state".
        """
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def _convert_obs(self, obs):
        return {"low_dim_state": obs.astype(np.float32)}


class D4RLEnvCompatibility(EnvCompatibility):
    """
    D4RL uses old gym environments. This Wrapper updates them to new
    gymnaisum environments with the updated API syntax.
    """

    def __init__(self, old_env: gym_old.Env, render_mode: Optional[str] = None):
        """Init.

        Args:
            old_env (gym_old.Env): an environment written with old_gym format.
            render_mode (Optional[str], optional): render mode. Defaults to None.
        """
        super().__init__(old_env, render_mode)

        # Assert the observation for old_env is box.
        assert isinstance(self.env.action_space, gym_old.spaces.Box)
        assert isinstance(self.env.observation_space, gym_old.spaces.Box)

        # Transform observation and action space from gym_old.Space into gymnasium.Space
        # Also force dtype to float32.
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=np.float32,
        )

    def get_normalized_score(self, score: float) -> float:
        """Get episode returns normalized against expert demos

        Args:
            score (float): episode returns

        Returns:
            (float): normalized episode returns
        """
        return self.env.get_normalized_score(score)


def _get_demo_fn(cfg: DictConfig, num_demos: int, demo_list: List):
    env = _make_env(cfg)
    d4rl_trajs, _ = get_traj_dataset(env)
    demo_list.extend(d4rl_trajs)
    env.close()


def _make_env(cfg: DictConfig) -> gym_old.Env:
    task_name = cfg.env.task_name

    # check task_name is supported
    task_env_name = task_name.split("-")[0]
    assert task_env_name in SUPPORTED_ENVS, f"{task_name} is not supported!"

    return gym_old.make(task_name)


class D4RLEnvFactory(EnvFactory):
    def _wrap_env(
        self, env: gym_old.Env | gym.Env, cfg: DictConfig, return_raw_spaces=False
    ):
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)
        # sanity check
        assert not cfg.pixels, "D4RL is state-only environment"
        if isinstance(env, gym_old.Env):
            # NOTE: For d4rl, the collected trajectory has 1 less step then
            # max_episode_steps.
            assert (
                cfg.env.episode_length == env.spec.max_episode_steps - 1
            ), "For D4RL, episode_length must be the same as the collected demo length."

        if isinstance(env, gym_old.Env):
            env = D4RLEnvCompatibility(env)

        env = ConvertObsToDict(env)
        if cfg.use_standardization:
            raise NotImplementedError("Not implemented and tested for D4RL")
        elif cfg.use_min_max_normalization:
            raise NotImplementedError("Not implemented and tested for D4RL")
        else:
            rescale_from_tanh_cls = RescaleFromTanh

        env = rescale_from_tanh_cls(env)
        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = ActionSequence(env, cfg.action_sequence)
        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, (action_space, observation_space)
        else:
            return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        """See base class for documentation."""
        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(_make_env(cfg), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        """See base class for documentation."""
        # NOTE: Assumes workspace always creates eval_env in the main thread
        env, (self._action_space, self._observation_space) = self._wrap_env(
            _make_env(cfg), cfg, return_raw_spaces=True
        )
        return env

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        """See base class for documentation."""
        # collect all demos
        manager = mp.Manager()
        mp_list = manager.list()
        p = mp.Process(
            target=_get_demo_fn,
            args=(
                cfg,
                num_demos,
                mp_list,
            ),
        )
        p.start()
        p.join()

        # Only extract num_demos from the full dataset
        all_demos = list(mp_list)
        if not math.isfinite(num_demos):
            num_demos = len(all_demos)

        if cfg.env.random_traj:
            self._raw_demos = random.sample(all_demos, num_demos)
        else:
            self._raw_demos = all_demos[:num_demos]

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = self._raw_demos

    def load_demos_into_replay(self, cfg: DictConfig, buffer):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._demos), self._action_space, self._observation_space
            ),
            cfg,
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)

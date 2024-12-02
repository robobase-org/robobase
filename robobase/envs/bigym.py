from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.action_modes import JointPositionActionMode
from robobase.utils import DemoEnv, add_demo_to_replay_buffer
from robobase.envs.utils.bigym_utils import TASK_MAP
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    RescaleFromTanhWithMinMax,
    OnehotTime,
    ActionSequence,
    AppendDemoInfo,
    FrameStack,
    ConcatDim,
    RecedingHorizonControl,
)
from omegaconf import DictConfig
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.action_modes import PelvisDof
import multiprocessing as mp
import logging
import numpy as np

from demonstrations.demo import DemoStep
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

from typing import List, Dict, Tuple, Callable
import copy

UNIT_TEST = False


def rescale_demo_actions(
    rescale_fn: Callable, demos: List[List[DemoStep]], cfg: DictConfig
):
    """Rescale actions in demonstrations to [-1, 1] Tanh space.
    This is because RoboBase assumes everything to be in [-1, 1] space.

    Args:
        rescale_fn: callable that takes info containing demo action and cfg and
            outputs the rescaled action
        demos: list of demo episodes whose actions are raw, i.e., not scaled
        cfg: Configs

    Returns:
        List[Demo]: list of demo episodes whose actions are rescaled
    """
    for demo in demos:
        for step in demo:
            info = step.info
            if "demo_action" in info:
                # Rescale demo actions
                info["demo_action"] = rescale_fn(info, cfg)
    return demos


def _task_name_to_env_class(task_name: str) -> type[BiGymEnv]:
    return TASK_MAP[task_name]


class BiGymEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg, demo_env=False, train=True, return_raw_spaces=False):
        # last two are grippers
        assert cfg.demos != 0
        assert cfg.action_repeat == 1

        action_space = copy.deepcopy(env.action_space)
        observation_space = copy.deepcopy(env.observation_space)

        env = RescaleFromTanhWithMinMax(
            env=env,
            action_stats=self._action_stats,
            min_max_margin=cfg.min_max_margin,
        )
        obs_stats = None
        if cfg.norm_obs:
            obs_stats = self._obs_stats

        # We normalize the low dimensional observations in the ConcatDim wrapper.
        # This is to be consistent with the original ACT implementation.
        env = ConcatDim(
            env,
            shape_length=1,
            dim=-1,
            new_name="low_dim_state",
            norm_obs=cfg.norm_obs,
            obs_stats=obs_stats,
            keys_to_ignore=["proprioception_floating_base_actions"],
        )
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)
        env = TimeLimit(
            env,
            cfg.env.episode_length // cfg.env.demo_down_sample_rate,
        )

        if not demo_env:
            if not train:
                env = RecedingHorizonControl(
                    env,
                    cfg.action_sequence,
                    cfg.env.episode_length // (cfg.env.demo_down_sample_rate),
                    cfg.execution_length,
                    temporal_ensemble=cfg.temporal_ensemble,
                    gain=cfg.temporal_ensemble_gain,
                )
            else:
                env = ActionSequence(
                    env,
                    cfg.action_sequence,
                )

        env = AppendDemoInfo(env)

        if return_raw_spaces:
            return env, action_space, observation_space
        else:
            return env

    def _create_env(self, cfg: DictConfig) -> BiGymEnv:
        bigym_class = _task_name_to_env_class(cfg.env.task_name)
        camera_configs = [
            CameraConfig(
                name=camera_name,
                rgb=True,
                depth=False,
                resolution=cfg.visual_observation_shape,
            )
            for camera_name in cfg.env.cameras
        ]

        if cfg.env.enable_all_floating_dof:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
                floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            )
        else:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
            )

        return bigym_class(
            render_mode=cfg.env.render_mode,
            action_mode=action_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs if cfg.pixels else [],
                proprioception=True,
                privileged_information=False if cfg.pixels else True,
            ),
            control_frequency=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.SyncVectorEnv
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    self._create_env(cfg),
                    cfg,
                    demo_env=False,
                    train=True,
                )
                for _ in range(cfg.num_train_envs)
            ],
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        env, self._action_space, self._observation_space = self._wrap_env(
            env=self._create_env(cfg),
            cfg=cfg,
            demo_env=False,
            train=False,
            return_raw_spaces=True,
        )
        return env

    def _get_demo_fn(self, cfg: DictConfig, num_demos: int):
        demos = []

        logging.info("Start to load demos.")
        env = self._create_env(cfg)

        demo_store = DemoStore()
        if np.isinf(num_demos):
            num_demos = -1

        demos = demo_store.get_demos(
            Metadata.from_env(env),
            amount=num_demos,
            frequency=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
        )

        for demo in demos:
            for ts in demo.timesteps:
                ts.observation = {
                    k: np.array(v, dtype=np.float32) for k, v in ts.observation.items()
                }

        env.close()
        logging.info("Finished loading demos.")
        return demos

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        demos = self._get_demo_fn(cfg, num_demos)
        self._raw_demos = demos
        self._action_stats = self._compute_action_stats(cfg, demos)
        self._obs_stats = self._compute_obs_stats(cfg, demos)

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        demo_list = [demo.timesteps for demo in self._raw_demos]
        demo_list = rescale_demo_actions(
            self._rescale_demo_action_helper, demo_list, cfg
        )
        self._demos = self._demo_to_steps(cfg, demo_list)

    def load_demos_into_replay(self, cfg: DictConfig, buffer, is_demo_buffer):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        
        if is_demo_buffer:
            # Filter successful demonstrations
            demos = []
            for i, demo in enumerate(self._demos):
                successful = (demo[0][-1]["demo"] == 1)
                if successful:
                    demos.append(demo)
                else:
                    print(f"Skipping failed demonstration {i}")
                    continue
        else:
            demos = self._demos

        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(demos), self._action_space, self._observation_space
            ),
            cfg,
            demo_env=True,
            train=False,
        )
        for _ in range(len(demos)):
            add_demo_to_replay_buffer(demo_env, buffer)

    def _demo_to_steps(
        self, cfg: DictConfig, demo_list: List[List[DemoStep]]
    ) -> List[DemoStep]:
        ret_demos = []

        for demo in demo_list:
            cur_demo = []
            last_timestep = False
            
            # Detect whether this demo is successful or not
            rewards = []
            for step in demo:
                reward = step.reward
                rewards.append(reward)
            successful_demo = sum(rewards) > 0.25
            
            for i, step in enumerate(demo):
                step.info.update({"demo": int(successful_demo)})
                if i == 0:
                    cur_demo.append((step.observation, step.info))
                else:
                    term, trunc = step.termination, step.truncation
                    reward = step.reward
                    if i == len(demo) - 1 or reward > 0:
                        if not (term or trunc):
                            term = False
                            trunc = True
                        last_timestep = True

                    cur_demo.append((step.observation, reward, term, trunc, step.info))
                if last_timestep:
                    break
            ret_demos.append(cur_demo)

        return ret_demos

    def _compute_action_stats(
        self, cfg: DictConfig, demos: List[List[DemoStep]]
    ) -> Dict:
        actions = []
        for demo in demos:
            for step in demo.timesteps:
                info = step.info
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)

        mean, std, gmax, gmin = self._get_gripper_action_stats(cfg)
        action_mean = np.hstack([np.mean(actions, 0)[:-2], mean, mean])
        action_std = np.hstack([np.std(actions, 0)[:-2], std, std])
        action_max = np.hstack([np.max(actions, 0)[:-2], gmax, gmax])
        action_min = np.hstack([np.min(actions, 0)[:-2], gmin, gmin])
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats

    def _compute_obs_stats(self, cfg: DictConfig, demos: List[List[DemoStep]]) -> Dict:
        obs = []
        for demo in demos:
            for step in demo.timesteps:
                obs.append(step.observation)

        keys = obs[0].keys()
        obs = {key: np.stack([o[key] for o in obs], axis=0) for key in keys}
        obs_mean = {key: np.mean(obs[key], 0) for key in keys}
        obs_std = {key: np.std(obs[key], 0) for key in keys}
        obs_min = {key: np.min(obs[key], 0) for key in keys}
        obs_max = {key: np.max(obs[key], 0) for key in keys}
        obs_stats = {
            "mean": obs_mean,
            "std": obs_std,
            "max": obs_max,
            "min": obs_min,
        }
        return obs_stats

    def _get_gripper_action_stats(
        self, cfg: DictConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if cfg.env.action_mode in ["absolute", "delta"]:
            return (0.5, 0.25, 1, 0)
        else:
            raise NotImplementedError("Unsupported action mode.")

    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        return RescaleFromTanhWithMinMax.transform_to_tanh(
            info["demo_action"],
            action_stats=self._action_stats,
            min_max_margin=cfg.min_max_margin,
        )

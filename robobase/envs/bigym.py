from enum import Enum

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.action_modes import ActionMode, JointPositionActionMode, TorqueActionMode
from robobase.utils import rescale_demo_actions, DemoEnv, add_demo_to_replay_buffer
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
from demonstrations.demo_store import DemoStore, DemoConverter
from demonstrations.utils import Metadata

from typing import List, Dict, Tuple
from pathlib import Path
import pickle
import copy

UNIT_TEST = False


class ActionModeType(Enum):
    TORQUE = "TORQUE"
    JOINT_POSITION = "JOINT_POSITION"


def _task_name_to_env_class(task_name: str) -> type[BiGymEnv]:
    return TASK_MAP[task_name]


def _create_action_mode(action_mode: str) -> ActionMode:
    if action_mode == ActionModeType.TORQUE.value:
        return TorqueActionMode()
    elif action_mode == ActionModeType.JOINT_POSITION.value:
        return JointPositionActionMode()


class BiGymEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg, demo_env=False, train=True, return_raw_spaces=False):
        # last two are grippers
        assert cfg.demos > 0
        assert cfg.action_repeat == 1

        action_space = copy.deepcopy(env.action_space)
        observation_space = copy.deepcopy(env.observation_space)

        env = RescaleFromTanhWithMinMax(
            action_stats=self._action_stats,
            min_max_margin=cfg.min_max_margin,
        )
        env = ConcatDim(
            env,
            shape_length=1,
            dim=-1,
            new_name="low_dim_state",
            keys_to_ignore=["proprioception_floating_base_actions"],
        )
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = TimeLimit(
            env, cfg.env.episode_length // cfg.demo_down_sample_rate,
        )
        env = ActionSequence(env, cfg.action_sequence)

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
            return env, (action_space, observation_space)
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
            ) for camera_name in cfg.env.cameras
        ]

        if cfg.env.enable_all_floating_dof:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
                floating_dofs=[
                    PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ
                ]
            )
        else:
            action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
            )
        
        return bigym_class(
            action_mode=action_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs if cfg.pixels else [],
                proprioception=True,
                privileged_information=False if cfg.pixels else True,
            )
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.AsyncVectorEnv
        kwargs = dict(context="fork")
        if UNIT_TEST:
            vec_env_class = gym.vector.SyncVectorEnv
            kwargs = dict()

        return vec_env_class(
            [
                lambda: self._wrap_env(
                    self._create_env(cfg),
                    demo_env=False,
                    train=True,
                ) for _ in range(cfg.num_train_envs)
            ], **kwargs
        )
    
    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        env, self._action_space, self._observation_space = self._wrap_env(
            env=self._create_env(cfg),
            demo_env=False,
            train=False,
            return_raw_spaces=True
        )
        return env

    def _get_demo_from_scratch(self, cfg: DictConfig, num_demos: int, mp_list: List) -> None:
        demos = []

        logging.info("Start to load demos.")
        env = self._create_env(cfg)

        demo_store = DemoStore()

        demos = demo_store.get_demos(Metadata.from_env(env), amount=-1)

        for demo in demos:
            for ts in demo.timesteps:
                ts.observation = {
                    k: np.array(v) for k, v in ts.observation.items()
                }
        
        env.close()
        logging.info("Finished loading demos.")
        mp_list.append(demos)
    
    def _get_demo_fn(self, cfg: DictConfig, num_demos: int, mp_list: List) -> None:
        dataset_root = cfg.env.dataset_root
        if dataset_root == "":
            dataset_root = Path.home() / ".bigym" / "cache"
        
        cache_dir = (
            dataset_root / cfg.env.env_name / cfg.env.task_name / cfg.env.action_mode / "pixels" if cfg.pixels else "low_dim_state"
        )
        cache_dir.makedirs(exist_ok=True)
        cache_path = cache_dir / "cache.pkl"

        if cache_path.exists():
            demos = pickle.load(cache_path.open("rb"))
            mp_list.append(demos)
            logging.info("Loaded demos from cache.")
        else:
            self._get_demo_from_scratch(cfg, num_demos, mp_list)
            demos = mp_list[0]
            pickle.dump(demos, cache_path.open("wb"), pickle.HIGHEST_PROTOCOL)

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        manager = mp.Manager()
        mp_list = manager.list()

        p = mp.Process(
            target=self._get_demo_fn,
            args=(cfg, num_demos, mp_list),
        )
        p.start()
        p.join()

        demos = mp_list[0]

        if num_demos < len(demos):
            demos = demos[:num_demos]
        
        self._raw_demos = [
            DemoConverter.decimate(
                demo,
                target_freq=CONTROL_FREQUENCY_MAX // cfg.env.demo_down_sample_rate,
            ) for demo in demos
        ]
        self._action_stats = self._compute_action_stats(cfg, demos)
    
    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        demo_list = [demo.timesteps for demo in self._raw_demos]
        demo_list = rescale_demo_actions(
            self._rescale_demo_action_helper, demo_list, cfg
        )
        self._demos = self._demo_to_steps(cfg, demo_list)
    
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
            demo_env=True,
            train=False,
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)


    def _demo_to_steps(self, cfg: DictConfig, demo_list: List[List[DemoStep]]) -> List[DemoStep]:
        ret_demos = []

        for demo in demo_list:
            cur_demo = []
            for i, step in enumerate(demo):
                step.info.update({"demo": 1})
                if i == 0:
                    cur_demo.append((step.observation, step.info))
                else:
                    term, trunc = step.termination, step.truncation
                    reward = step.reward
                    if i == len(demo) - 1:
                        if not (term or trunc):
                            term = False
                            trunc = True
                        
                        reward = 1
                    
                    cur_demo.append(
                        (
                            step.observation,
                            reward,
                            term,
                            trunc,
                            step.info
                        )
                    )
            ret_demos.append(cur_demo)

        return ret_demos
    
    def _compute_action_stats(self, cfg: DictConfig, demos: List[List[DemoStep]]) -> Dict:
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
    
    def _get_gripper_action_stats(self, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (0.5, 1, 1, 0)
    
    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        return RescaleFromTanhWithMinMax.transform_to_tanh(
            info["demo_action"], action_stats=self._action_stats, min_max_margin=cfg.min_max_margin
        )

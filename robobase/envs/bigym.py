from enum import Enum

from bigym.bigym_env import BiGymEnv
from bigym.action_modes import ActionMode, JointPositionActionMode, TorqueActionMode
from robobase.envs.utils.bigym_utils import TASK_MAP
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    RescaleFromTanh,
    OnehotTime,
    ActionSequence,
    AppendDemoInfo,
    FrameStack,
    ConcatDim,
)
from omegaconf import DictConfig

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
    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        env = ConcatDim(env, 1, -1, "low_dim_state")
        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)
        env = FrameStack(env, cfg.frame_stack)
        env = ActionSequence(env, cfg.action_sequence)
        env = AppendDemoInfo(env)
        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.AsyncVectorEnv
        kwargs = dict(context="fork")
        if UNIT_TEST:
            vec_env_class = gym.vector.SyncVectorEnv
            kwargs = dict()
        bygym_class = _task_name_to_env_class(cfg.env.task_name)
        action_mode = _create_action_mode(cfg.env.action_mode)
        cameras = cfg.env.cameras if cfg.pixels else None
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    bygym_class(
                        action_mode=action_mode,
                        cameras=cameras,
                        camera_resolution=cfg.visual_observation_shape,
                        render_mode="rgb_array",
                    ),
                    cfg,
                )
                for _ in range(cfg.num_train_envs)
            ],
            **kwargs,
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        bygym_class = _task_name_to_env_class(cfg.env.task_name)
        action_mode = _create_action_mode(cfg.env.action_mode)
        cameras = cfg.env.cameras if cfg.pixels else None
        return self._wrap_env(
            bygym_class(
                action_mode=action_mode,
                cameras=cameras,
                camera_resolution=cfg.visual_observation_shape,
                render_mode="rgb_array",
            ),
            cfg,
        )

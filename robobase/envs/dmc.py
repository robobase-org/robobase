import sys

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from dm_control import manipulation, suite
from dm_control.suite.wrappers import pixels
from dm_env import specs
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig

from robobase.envs.env import EnvFactory
from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    ActionSequence,
)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "darwin" else "egl"
UNIT_TEST = False


def _convert_dm_control_to_gym_space(dm_control_space, dtype=None, **kwargs):
    """Convert dm_control space to gym space."""
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(
            low=dm_control_space.minimum,
            high=dm_control_space.maximum,
            dtype=dtype or dm_control_space.dtype,
        )
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(
        dm_control_space, specs.BoundedArray
    ):
        space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=dtype or dm_control_space.dtype,
        )
        return space
    elif isinstance(dm_control_space, dict):
        kwargs.update(
            {
                key: _convert_dm_control_to_gym_space(value, dtype=dtype)
                for key, value in dm_control_space.items()
            }
        )
        space = spaces.Dict(kwargs)
        return space


class DMC(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        task_name,
        from_pixels=False,
        action_repeat: int = 1,
        visual_observation_shape: tuple[int, int] = (84, 84),
        render_mode: str = None,
    ):
        domain, task = task_name.split("_", 1)
        self._from_pixels = from_pixels
        self._action_repeat = action_repeat
        self._viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode
        # overwrite cup to ball_in_cup
        domain = dict(cup="ball_in_cup").get(domain, domain)
        camera_id = 0
        if (domain, task) in suite.ALL_TASKS:
            self._dmc_env = suite.load(domain, task, visualize_reward=False)
            self._pixels_key = "pixels"
            # zoom in camera for quadruped
            camera_id = dict(quadruped=2).get(domain, 0)
        else:
            name = f"{domain}_{task}_vision"
            self._dmc_env = manipulation.load(name)
            self._pixels_key = "front_close"

        height, width = visual_observation_shape
        self._render_kwargs = dict(height=height, width=width, camera_id=camera_id)
        if from_pixels:
            image_shape = [3, height, width]
            self._dmc_env = pixels.Wrapper(
                self._dmc_env, pixels_only=True, render_kwargs=self._render_kwargs
            )
            obs_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
            self.observation_space = spaces.Dict({"rgb": obs_space})
        else:
            obs_spec = self._dmc_env.observation_spec()
            obs_space = _convert_dm_control_to_gym_space(obs_spec, dtype=np.float32)
            obs_space = spaces.flatten_space(obs_space)
            self.observation_space = spaces.Dict({"low_dim_state": obs_space})
        self.action_space = _convert_dm_control_to_gym_space(
            self._dmc_env.action_spec(), dtype=np.float32
        )

    def _get_obs(self, timestep):
        obs = timestep.observation
        ret_obs = {}
        if self._from_pixels:
            ret_obs["rgb"] = (
                timestep.observation[self._pixels_key].transpose([2, 0, 1]).copy()
            )
        else:
            ret_obs["low_dim_state"] = self._flatten_obs(obs)
        return ret_obs

    def _flatten_obs(self, observation):
        obs_pieces = []
        for v in observation.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=0).astype(np.float32)
        return obs

    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            ts = self._dmc_env.step(action)
            reward += ts.reward
            if ts.last():
                break
        # See https://github.com/google-deepmind/dm_control/blob/f2f0e2333d8bd82c0b6ba83628fe44c2bcc94ef5/dm_control/rl/control.py#L115C18-L115C29
        truncated = ts.last() and ts.discount == 1.0
        terminal = ts.last() and not truncated
        # Either both false or xor.
        assert not np.any(
            terminal and truncated
        ), "Can't be both terminal and truncated."
        return self._get_obs(ts), reward, terminal, truncated, {}

    def reset(self, seed=None, options=None):
        self.action_space.seed(seed)
        self._dmc_env.task.random.seed(seed)
        timestep = self._dmc_env.reset()
        return self._get_obs(timestep), {}

    def render(self):
        img = self._dmc_env.physics.render(
            width=self._render_kwargs["width"],
            height=self._render_kwargs["height"],
            camera_id=self._render_kwargs["camera_id"],
        )
        if self._render_mode == "rgb_array":
            return img.astype(np.uint8)
        elif self._render_mode == "human":
            from PIL import Image

            return Image.fromarray(img)
        else:
            raise NotImplementedError(f"`{self._render_mode}` mode is not implemented")

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return self._dmc_env.close()


class DMCEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        if cfg.env.episode_length != 1000:
            # Used in unit tests.
            env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(
                env, cfg.env.episode_length // cfg.action_repeat
            )  # Time limits are handles by DMC
        env = ActionSequence(env, cfg.action_sequence)
        env = FrameStack(env, cfg.frame_stack)
        return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.AsyncVectorEnv
        kwargs = dict(context="spawn")
        if UNIT_TEST:
            vec_env_class = gym.vector.SyncVectorEnv
            kwargs = dict()
        else:
            assert cfg.env.episode_length == 1000, "DMC episode length must be 1000."
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    DMC(
                        cfg.env.task_name,
                        cfg.pixels,
                        cfg.action_repeat,
                        cfg.visual_observation_shape,
                        "rgb_array",
                    ),
                    cfg,
                )
                for _ in range(cfg.num_train_envs)
            ],
            **kwargs,
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        return self._wrap_env(
            DMC(
                cfg.env.task_name,
                cfg.pixels,
                cfg.action_repeat,
                cfg.visual_observation_shape,
                "rgb_array",
            ),
            cfg,
        )

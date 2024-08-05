import os
import time
import importlib
import warnings
from typing import List
from enum import Enum
from functools import partial
import copy

from omegaconf import DictConfig
from pyrep.const import RenderMode
from pyrep.objects import Dummy, VisionSensor
from pyrep.errors import ConfigurationPathError, IKError
from gymnasium.wrappers import TimeLimit

from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    RescaleFromTanhEEPose,
    RescaleFromTanhWithStandardization,
    RescaleFromTanhWithMinMax,
    ActionSequence,
    RecedingHorizonControl,
    AppendDemoInfo,
)
from robobase.utils import (
    DemoStep,
    observations_to_timesteps,
    add_demo_to_replay_buffer,
)
from robobase.utils import (
    observations_to_action_with_onehot_gripper,
    observations_to_action_with_onehot_gripper_nbp,
    rescale_demo_actions,
)
from robobase.envs.env import EnvFactory, Demo, DemoEnv
import multiprocessing as mp

try:
    from rlbench import ObservationConfig, Environment, CameraConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.backend.observation import Observation
    from rlbench.backend.exceptions import InvalidActionError
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionModeType(Enum):
    END_EFFECTOR_POSE = 1
    JOINT_POSITION = 2


ROBOT_STATE_KEYS = [
    "joint_velocities",
    "joint_positions",
    "joint_forces",
    "gripper_open",
    "gripper_pose",
    "gripper_matrix",
    "gripper_joint_positions",
    "gripper_touch_forces",
    "task_low_dim_state",
    "misc",
]

TASK_TO_LOW_DIM_SIM = {
    "reach_target": 3,
    "pick_and_lift": 6,
    "take_lid_off_saucepan": 7,
}


def _extract_obs(obs: Observation, observation_config, robot_state_keys: dict = None):
    obs_dict = vars(obs)

    if robot_state_keys is not None:
        obs_dict = {
            k: None if k in robot_state_keys else v for k, v in obs_dict.items()
        }
        obs = Observation(**obs_dict)
    robot_state = obs.get_low_dim_data()

    obs_dict = {
        k: v for k, v in obs_dict.items() if v is not None and k not in ROBOT_STATE_KEYS
    }

    obs_dict = {
        k: v.transpose((2, 0, 1)) if v.ndim == 3 else np.expand_dims(v, 0)
        for k, v in obs_dict.items()
    }
    obs_dict["low_dim_state"] = np.array(robot_state, dtype=np.float32)
    for k, v in [(k, v) for k, v in obs_dict.items() if "point_cloud" in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.left_shoulder_camera, "left_shoulder"),
        (observation_config.right_shoulder_camera, "right_shoulder"),
        (observation_config.front_camera, "front"),
        (observation_config.wrist_camera, "wrist"),
        (observation_config.overhead_camera, "overhead"),
    ]:
        if config.point_cloud:
            obs_dict["%s_camera_extrinsics" % name] = obs.misc[
                "%s_camera_extrinsics" % name
            ]
            obs_dict["%s_camera_intrinsics" % name] = obs.misc[
                "%s_camera_intrinsics" % name
            ]
    return obs_dict


def _get_cam_observation_elements(camera: CameraConfig, prefix: str):
    space_dict = {}
    img_s = camera.image_size
    if camera.rgb:
        space_dict["%s_rgb" % prefix] = spaces.Box(
            0, 255, shape=(3,) + img_s, dtype=np.uint8
        )
    if camera.point_cloud:
        space_dict["%s_point_cloud" % prefix] = spaces.Box(
            -np.inf, np.inf, shape=(3,) + img_s, dtype=np.float32
        )
        space_dict["%s_camera_extrinsics" % prefix] = spaces.Box(
            -np.inf, np.inf, shape=(4, 4), dtype=np.float32
        )
        space_dict["%s_camera_intrinsics" % prefix] = spaces.Box(
            -np.inf, np.inf, shape=(3, 3), dtype=np.float32
        )
    if camera.depth:
        space_dict["%s_depth" % prefix] = spaces.Box(
            0, np.inf, shape=(1,) + img_s, dtype=np.float32
        )
    if camera.mask:
        raise NotImplementedError()
    return space_dict


def _observation_config_to_gym_space(observation_config, task_name: str) -> spaces.Dict:
    space_dict = {}
    robot_state_len = 0
    if observation_config.joint_velocities:
        robot_state_len += 7
    if observation_config.joint_positions:
        robot_state_len += 7
    if observation_config.joint_forces:
        robot_state_len += 7
    if observation_config.gripper_open:
        robot_state_len += 1
    if observation_config.gripper_pose:
        robot_state_len += 7
    if observation_config.gripper_joint_positions:
        robot_state_len += 2
    if observation_config.gripper_touch_forces:
        robot_state_len += 2
    if observation_config.task_low_dim_state:
        robot_state_len += TASK_TO_LOW_DIM_SIM[task_name]
    if robot_state_len > 0:
        space_dict["low_dim_state"] = spaces.Box(
            -np.inf, np.inf, shape=(robot_state_len,), dtype=np.float32
        )
    for cam, name in [
        (observation_config.left_shoulder_camera, "left_shoulder"),
        (observation_config.right_shoulder_camera, "right_shoulder"),
        (observation_config.front_camera, "front"),
        (observation_config.wrist_camera, "wrist"),
        (observation_config.overhead_camera, "overhead"),
    ]:
        space_dict.update(_get_cam_observation_elements(cam, name))
    return spaces.Dict(space_dict)


def _name_to_task_class(task_file: str):
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    try:
        mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise ValueError(
            "The env file '%s' does not exist or cannot be compiled." % name
        ) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise ValueError(
            "Cannot find the class name '%s' in the file '%s'." % (class_name, name)
        ) from e
    return task_class


def _convert_rlbench_demos_for_loading(
    raw_demos, observation_config, robot_state_keys: dict = None
) -> List[List[DemoStep]]:
    """Converts demos generated in rlbench to the common DemoStep format.

    Args:
        raw_demos: raw demos generated with rlbench.

    Returns:
        List[List[DemoStep]]: demos converted to DemoSteps ready for
            augmentation and loading.
    """
    converted_demos = []
    for demo in raw_demos:
        converted_demo = []
        for timestep in demo:
            converted_demo.append(
                DemoStep(
                    timestep.joint_positions,
                    timestep.gripper_open,
                    _extract_obs(timestep, observation_config, robot_state_keys),
                    timestep.gripper_matrix,
                    timestep.misc,
                )
            )
        converted_demos.append(converted_demo)
    return converted_demos


def keypoint_discovery(
    demo: Demo, stopping_delta: float = 0.1, method: str = "heuristic"
) -> list[int]:
    """Discover next-best-pose keypoints in a demonstration based on specified method.

    Args:
        demo: A demonstration represented as a list of Observation objects.
        stopping_delta: Tolerance for considering joint
            velocities as "stopped". Defaults to 0.1.
        method: The method for discovering keypoints.
            - "heuristic": Uses a heuristic approach.
            - "random": Randomly selects keypoints.
            - "fixed_interval": Selects keypoints at fixed intervals.
            Defaults to "heuristic".

    Returns:
        List of indices representing the discovered keypoints in the demonstration.
    """
    episode_keypoints = []

    if method == "heuristic":
        # Gripper open is the first low_dim_state index.
        prev_gripper_open = demo[0][0]["low_dim_state"][0]
        stopped_buffer = 0
        for i, data in enumerate(demo):
            # Data is a tuple
            # First index is (obs, info)
            # Other indices are (obs, reward, termination, truncation, info, action)
            state = data[0]["low_dim_state"]

            # Check if the current time step is not the second-to-last step in the
            # demonstration
            next_is_not_final = i == (len(demo) - 2)
            # Check if the gripper state hasn't changed for the past two time steps
            # and one time step ahead
            # NOTE: Is using np.all_close safer?
            gripper_state_no_change = i < (len(demo) - 2) and (
                demo[i + 1][0]["low_dim_state"][0]
                == state[0]
                == demo[i - 1][0]["low_dim_state"][0]
                == demo[i - 2][0]["low_dim_state"][0]
            )
            # Check if all joint velocities are close to zero within the given tolerance
            small_delta = np.allclose(state[1:8], 0.0, atol=stopping_delta)

            # Check if the system is considered stopped based on various conditions
            stopped = (
                stopped_buffer <= 0
                and small_delta
                and (not next_is_not_final)
                and gripper_state_no_change
            )

            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (state[0] != prev_gripper_open or last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = state[0]
        if (
            len(episode_keypoints) > 1
            and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
        ):
            episode_keypoints.pop(-2)
        return episode_keypoints

    elif method == "random":
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo._observations)), size=20, replace=False
        )
        episode_keypoints.sort()
        return episode_keypoints

    elif method == "fixed_interval":
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo._observations) // 20
        for i in range(0, len(demo._observations), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


class RLBenchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        task_name: str,
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        action_mode_type: ActionModeType = ActionModeType.JOINT_POSITION,
        arm_max_velocity: float = 1.0,
        arm_max_acceleration: float = 4.0,
        dataset_root: str = "",
        renderer: str = "opengl",
        headless: bool = True,
        render_mode: str = None,
    ):
        self._task_name = task_name
        self._observation_config = observation_config
        self._action_mode = action_mode
        self._action_mode_type = action_mode_type
        self._arm_max_velocity = arm_max_velocity
        self._arm_max_acceleration = arm_max_acceleration
        self._dataset_root = dataset_root
        self._headless = headless
        self._rlbench_env = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_space = _observation_config_to_gym_space(
            observation_config, task_name
        )
        minimum, maximum = action_mode.action_bounds()
        self.action_space = spaces.Box(
            minimum, maximum, shape=maximum.shape, dtype=maximum.dtype
        )
        if renderer == "opengl":
            self.renderer = RenderMode.OPENGL
        elif renderer == "opengl3":
            self.renderer = RenderMode.OPENGL3
        else:
            raise ValueError(self.renderer)

    def _launch(self):
        task_class = _name_to_task_class(self._task_name)
        self._rlbench_env = Environment(
            action_mode=self._action_mode,
            obs_config=self._observation_config,
            dataset_root=self._dataset_root,
            headless=self._headless,
            arm_max_velocity=self._arm_max_velocity,
            arm_max_acceleration=self._arm_max_acceleration,
        )
        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(task_class)
        if self.render_mode is not None:
            self._add_video_camera()

    def _add_video_camera(self):
        cam_placeholder = Dummy("cam_cinematic_placeholder")
        self._cam = VisionSensor.create([320, 192], explicit_handling=True)
        self._cam.set_pose(cam_placeholder.get_pose())
        self._cam.set_render_mode(self.renderer)

    def _render_frame(self) -> np.ndarray:
        self._cam.handle_explicitly()
        frame = self._cam.capture_rgb()
        frame = np.clip((frame * 255.0).astype(np.uint8), 0, 255)
        return frame

    def _ee_pose_step(self, action):
        # Catch exception and terminate if unable to reach pose
        try:
            rlb_obs, reward, term = self._task.step(action)
        except (IKError, ConfigurationPathError, InvalidActionError):
            rlb_obs = Observation(
                *([None] * 30)
            )  # Set to empty observation as not used on terminate step
            rlb_obs.task_low_dim_state = np.zeros(8)
            reward = 0.0
            term = True

        return rlb_obs, reward, term

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def step(self, action):
        if self._action_mode_type == ActionModeType.END_EFFECTOR_POSE:
            rlb_obs, reward, term = self._ee_pose_step(action)
        else:
            rlb_obs, reward, term = self._task.step(action)
        obs = _extract_obs(rlb_obs, self._observation_config)
        return obs, reward, term, False, {"demo": 0, "task_success": int(reward > 0)}

    def reset(self, seed=None, options=None, robot_state_keys: dict = None):
        super().reset(seed=seed)
        if self._rlbench_env is None:
            self._launch()
        _, rlb_obs = self._task.reset()

        obs = _extract_obs(rlb_obs, self._observation_config, robot_state_keys)
        return obs, {"demo": 0}

    def close(self):
        if self._rlbench_env is not None:
            self._rlbench_env.shutdown()

    def get_nbp_demos(self, demos):
        demo_augmentation = True
        demo_augmentation_every_n = 5

        nbp_demos = []
        for idx_demo, demo in enumerate(demos):
            episode_keypoints = keypoint_discovery(demo)
            for idx, step in enumerate(demo):
                if not demo_augmentation and idx > 0:
                    break
                if idx % demo_augmentation_every_n != 0:
                    continue

                nbp_demo = []
                nbp_demo.append(step)

                # If our starting point is past one of the keypoints, then remove it
                while len(episode_keypoints) > 0 and idx >= episode_keypoints[0]:
                    episode_keypoints = episode_keypoints[1:]

                if len(episode_keypoints) == 0:
                    break

                for episode_keypoint in episode_keypoints:
                    keypoint = copy.deepcopy(demo[episode_keypoint])
                    nbp_demo.append(keypoint)

                nbp_demos.append(nbp_demo)

        return nbp_demos

    def get_demos(self, num_demos: int, robot_state_keys: dict = None) -> List[Demo]:
        live_demos = not self._dataset_root
        if live_demos:
            warnings.warn(
                "dataset_root was not defined. Generating live demos. "
                "This may take a while..."
            )
        raw_demos = self._task.get_demos(num_demos, live_demos=live_demos)
        match self._action_mode_type:
            case ActionModeType.END_EFFECTOR_POSE:
                raw_demos = self.get_nbp_demos(raw_demos)
                action_func = observations_to_action_with_onehot_gripper_nbp
            case ActionModeType.JOINT_POSITION:
                action_func = observations_to_action_with_onehot_gripper

                # NOTE: Check there is a misc["joint_position_action"]
                is_joint_position_action_included = False
                for obs in raw_demos[0]:
                    if "joint_position_action" in obs.misc:
                        is_joint_position_action_included = True
                        break
                assert is_joint_position_action_included, (
                    "`joint_position_action` is not in obs.misc, "
                    "which could severely affect performance. Please use the "
                    "latest version of PyRep and RLBench for collecting demos."
                )

        demos_to_load = _convert_rlbench_demos_for_loading(
            raw_demos,
            self._observation_config,
            robot_state_keys=robot_state_keys,
        )
        # Process the demos using the selected action function
        loaded_demos = []
        for demo in demos_to_load:
            loaded_demos += observations_to_timesteps(
                demo, self.action_space, skipping=False, obs_to_act_func=action_func
            )
        return loaded_demos


def _make_obs_config(cfg: DictConfig):
    pixels = cfg.pixels

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(False)
    obs_config.set_all_high_dim(False)
    obs_config.gripper_open = True
    obs_config.joint_positions = True

    if pixels:
        if cfg.env.renderer == "opengl":
            renderer = RenderMode.OPENGL
        elif cfg.env.renderer == "opengl3":
            renderer = RenderMode.OPENGL3
        else:
            raise ValueError(cfg.env.renderer)

        for camera in cfg.env.cameras:
            camera_config = getattr(obs_config, f"{camera}_camera")
            setattr(camera_config, "rgb", True)
            setattr(camera_config, "image_size", cfg.visual_observation_shape)
            setattr(camera_config, "render_mode", renderer)
            setattr(obs_config, f"{camera}_camera", camera_config)
    else:
        obs_config.task_low_dim_state = True

    return obs_config


def _get_demo_fn(cfg, num_demos, demo_list):
    obs_config = _make_obs_config(cfg)
    obs_config_demo = copy.deepcopy(obs_config)

    # RLBench demos are all saved in same action mode (joint).
    # For conversion to an alternate action mode, additional
    # info may be required. ROBOT_STATE_KEYS is altered to
    # reflect this and ensure low_dim_state is consitent
    # for demo and rollout steps.

    match ActionModeType[cfg.env.action_mode]:
        case ActionModeType.END_EFFECTOR_POSE:
            obs_config_demo.joint_velocities = True
            obs_config_demo.gripper_matrix = True

        case ActionModeType.JOINT_POSITION:
            pass

        case _:
            raise ValueError(f"Unsupported action mode type: {cfg.env.action_mode}")

    # Get common true attribute in both configs and alter ROBOT_STATE_KEYS
    common_true = [
        attr_name
        for attr_name in dir(obs_config_demo)
        if isinstance(getattr(obs_config_demo, attr_name), bool)
        and getattr(obs_config_demo, attr_name)
        and getattr(obs_config, attr_name)
        # if "camera" not in attr_name
    ]
    demo_state_keys = copy.deepcopy(ROBOT_STATE_KEYS)
    for attr in common_true:
        demo_state_keys.remove(attr)

    rlb_env = _make_env(cfg, obs_config_demo)
    rlb_env.reset(robot_state_keys=demo_state_keys)

    demos = rlb_env.get_demos(num_demos, robot_state_keys=demo_state_keys)
    demo_list.extend(demos)
    rlb_env.close()


def _get_spaces(cfg, space_list):
    obs_config = _make_obs_config(cfg)
    rlb_env = _make_env(cfg, obs_config)
    space_list.append(
        (rlb_env.unwrapped.observation_space, rlb_env.unwrapped.action_space)
    )
    rlb_env.close()


def _make_env(cfg: DictConfig, obs_config: dict):
    # NOTE: Completely random initialization
    # TODO: Can we make this deterministic based on cfg.seed?
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    task_name = cfg.env.task_name
    action_mode = _get_action_mode(ActionModeType[cfg.env.action_mode])

    return RLBenchEnv(
        task_name,
        obs_config,
        action_mode,
        action_mode_type=ActionModeType[cfg.env.action_mode],
        arm_max_velocity=cfg.env.arm_max_velocity,
        arm_max_acceleration=cfg.env.arm_max_acceleration,
        dataset_root=cfg.env.dataset_root,
        render_mode="rgb_array",
    )


def _get_action_mode(action_mode_type: ActionModeType):
    match action_mode_type:
        case ActionModeType.END_EFFECTOR_POSE:

            class CustomMoveArmThenGripper(MoveArmThenGripper):
                def action_bounds(
                    self,
                ):  ## x,y,z,quat,gripper -> 8. Limited by rlbench scene workspace
                    return (
                        np.array(
                            [-0.3, -0.5, 0.6] + 3 * [-1.0] + 2 * [0.0],
                            dtype=np.float32,
                        ),
                        np.array([0.7, 0.5, 1.6] + 4 * [1.0] + [1.0], dtype=np.float32),
                    )

            action_mode = CustomMoveArmThenGripper(
                EndEffectorPoseViaPlanning(), Discrete()
            )
        case ActionModeType.JOINT_POSITION:

            class CustomMoveArmThenGripper(MoveArmThenGripper):
                def action_bounds(self):
                    return (
                        np.array(7 * [-0.2] + [0.0], dtype=np.float32),
                        np.array(7 * [0.2] + [1.0], dtype=np.float32),
                    )

            action_mode = CustomMoveArmThenGripper(JointPosition(False), Discrete())

    return action_mode


class RLBenchEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg, return_raw_spaces=False, demo_env=False):
        if return_raw_spaces:
            action_space = copy.deepcopy(env.action_space)
            observation_space = copy.deepcopy(env.observation_space)
        if ActionModeType[cfg.env.action_mode] == ActionModeType.END_EFFECTOR_POSE:
            rescale_from_tanh_cls = RescaleFromTanhEEPose
        else:
            assert not (
                cfg.use_standardization and cfg.use_min_max_normalization
            ), "You can't use both standardization and min/max normalization."
            if cfg.use_standardization:
                # Use demo-based standardization for actions
                assert cfg.demos > 0
                rescale_from_tanh_cls = partial(
                    RescaleFromTanhWithStandardization,
                    action_stats=self._action_stats,
                )
            elif cfg.use_min_max_normalization:
                # Use demo-based min/max normalization for actions
                assert cfg.demos > 0
                rescale_from_tanh_cls = partial(
                    RescaleFromTanhWithMinMax,
                    action_stats=self._action_stats,
                    min_max_margin=cfg.min_max_margin,
                )
            else:
                rescale_from_tanh_cls = RescaleFromTanh
        env = rescale_from_tanh_cls(env)
        env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(env, cfg.env.episode_length)

        # As RoboBase replay buffer always store single-step transitions. demo-env
        # should ignores action sequence and frame stack wrapper.
        if not demo_env:
            env = FrameStack(env, cfg.frame_stack)

            # If action_sequence length and execution length are the same, we do not
            # use receding horizon wrapper.
            # NOTE: for RL, action_sequence == execution_length == 1, so
            #       RecedingHorizonControl won't be enabled.
            if cfg.action_sequence == cfg.execution_length:
                env = ActionSequence(
                    env,
                    cfg.action_sequence,
                )
            else:
                env = RecedingHorizonControl(
                    env,
                    cfg.action_sequence,
                    cfg.env.episode_length,
                    cfg.execution_length,
                    cfg.temporal_ensemble,
                    cfg.temporal_ensemble_gain,
                )

        env = AppendDemoInfo(env)
        if return_raw_spaces:
            return env, (action_space, observation_space)
        else:
            return env

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        obs_config = _make_obs_config(cfg)

        return gym.vector.AsyncVectorEnv(
            [
                lambda: self._wrap_env(_make_env(cfg, obs_config), cfg)
                for _ in range(cfg.num_train_envs)
            ]
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        obs_config = _make_obs_config(cfg)
        # NOTE: Assumes workspace always creates eval_env in the main thread
        env, (self._action_space, self._observation_space) = self._wrap_env(
            _make_env(cfg, obs_config), cfg, return_raw_spaces=True
        )
        return env

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        """See base class for documentation."""

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

        self._raw_demos = list(mp_list)

        # Compute action statistics for demo-based rescaling, e.g., standardization
        self._action_stats = self._compute_action_stats(self._raw_demos)

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = rescale_demo_actions(
            self._rescale_demo_action_helper, self._raw_demos, cfg
        )

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
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)

    def _rescale_demo_action_helper(self, info, cfg: DictConfig):
        match ActionModeType[cfg.env.action_mode]:
            case ActionModeType.END_EFFECTOR_POSE:
                return RescaleFromTanhEEPose.transform_to_tanh(
                    info["demo_action"], self._action_space
                )
            case ActionModeType.JOINT_POSITION:
                if cfg.use_standardization:
                    return RescaleFromTanhWithStandardization.transform_to_tanh(
                        info["demo_action"], action_stats=self._action_stats
                    )
                elif cfg.use_min_max_normalization:
                    return RescaleFromTanhWithMinMax.transform_to_tanh(
                        info["demo_action"],
                        action_stats=self._action_stats,
                        min_max_margin=cfg.min_max_margin,
                    )
                else:
                    return RescaleFromTanh.transform_to_tanh(
                        info["demo_action"], self._action_space
                    )

    def _compute_action_stats(self, demos: List[List[DemoStep]]):
        """Compute statistics from demonstration actions, which could be useful for
        users that want to set action space based on demo action statistics.

        Args:
            demos: list of demo episodes

        Returns:
            Dict[str, np.ndarray]: a dictionary of numpy arrays that contain action
            statistics (i.e., mean, std, max, and min)
        """
        actions = []
        for demo in demos:
            for step in demo:
                *_, info = step
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)

        # Gripper one-hot action's stats are hard-coded
        action_mean = np.hstack([np.mean(actions, 0)[:-1], 1 / 2])
        action_std = np.hstack([np.std(actions, 0)[:-1], 1 / 6])
        action_max = np.hstack([np.max(actions, 0)[:-1], 1])
        action_min = np.hstack([np.min(actions, 0)[:-1], 0])
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats

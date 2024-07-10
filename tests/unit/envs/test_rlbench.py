import pytest

from rlbench.observation_config import ObservationConfig
from robobase.envs.env import Demo
from robobase.envs.rlbench import RLBenchEnv, keypoint_discovery
from rlbench.action_modes.action_mode import JointPositionActionMode


@pytest.fixture
def demo():
    """Get demo from rlbench."""
    cameras = ["wrist"]
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)
    obs_config.gripper_open = True
    obs_config.joint_positions = True
    for camera in cameras:
        camera_config = getattr(obs_config, f"{camera}_camera")
        camera_config.rgb = True
        camera_config.depth = False
        camera_config.point_cloud = False
        camera_config.mask = False

    env = RLBenchEnv(
        task_name="reach_target",
        observation_config=obs_config,
        action_mode=JointPositionActionMode(),
        headless=True,
    )
    _, _ = env.reset()
    demos = env.get_demos(1)
    env.close()
    return demos[0]


def test_keypoint_discovery(demo):
    keypoint_indices = keypoint_discovery(demo, method="heuristic")
    # 0th index is not included in keypoint_indices
    nbp_demo = Demo([demo[i] for i in [0] + keypoint_indices])
    assert len(keypoint_indices) < len(demo)
    # Last index should be in keypoint_indices
    assert len(demo) - 1 in keypoint_indices
    # Check reward of last keypoint
    assert nbp_demo[-1][1] > 0

    # TODO: Add tests for other keypoint discovery methods (random, fixed_interval)

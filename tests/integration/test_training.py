import csv
import tempfile
import time
from pathlib import Path
import multiprocessing

import numpy as np
import pytest

from hydra import compose, initialize
from omegaconf import OmegaConf

from robobase.workspace import Workspace

EXP_NAME = "test_experiment"
COMMON_HYDRA_OVERRIDES = [
    "num_train_frames=100000",
    "replay_size_before_train=1000",
    "save_csv=true",
    "log_eval_video=false",
    "num_train_envs=1",
    "seed=1",
    f"experiment_name={EXP_NAME}",
]
PIXEL_METHODS = ["drqv2", "alix"]
STATE_METHODS = ["drqv2"]


def run_cmd(hydra_overrides: list[str], target_reward: float, result_queue):
    try:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdirname:
            with initialize(
                version_base=None,
                config_path="../../robobase/cfgs",
                job_name="test_app",
            ):
                hydra_overrides.append(f"replay.save_dir={tmpdirname}/replay")
                tmp_dir = Path(tmpdirname)
                cfg = compose(config_name="robobase_config", overrides=hydra_overrides)
                print(OmegaConf.to_yaml(cfg))
                workspace = Workspace(cfg, work_dir=tmp_dir)
                start_time = time.monotonic()
                workspace.train()
                print(
                    "Train time: ", (time.monotonic() - start_time) / 60.0, " minutes"
                )
                csv_log_dir = Path(tmp_dir, "eval.csv")
                with csv_log_dir.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    returns = [row["episode_reward"] for row in reader]
                sorted_rewards = np.sort(np.array(returns[-20:]).astype(float))
                # Check the top N rewards (reduces test flakiness)
                print(np.mean(sorted_rewards[-5:]))
                assert np.mean(sorted_rewards[-5:]) > target_reward
    except Exception as e:  # pylint: disable=try-except-raise
        result_queue.put(e)
    finally:
        # Sleep for 5 secs to clean memory from previous process
        # Next test will fail without this when it's initialised too fast
        time.sleep(5)


@pytest.mark.parametrize("method", STATE_METHODS)
def test_dmc_cartpole_no_pixels(method: str):
    result_queue = multiprocessing.Queue()
    # Create a multiprocessing.Process and pass the result queue as an argument
    process = multiprocessing.Process(
        target=run_cmd,
        args=(
            COMMON_HYDRA_OVERRIDES
            + [
                f"method={method}",
                "env=dmc/cartpole_balance",
                "pixels=false",
                "frame_stack=3",
                "method.use_augmentation=true",
                "replay.size=200000",
            ],
            800,
            result_queue,
        ),
    )

    # Start the process
    process.start()
    # Wait for the process to finish
    process.join()
    # Check if the process terminated successfully (exit code 0) or not
    if process.exitcode != 0:
        raise result_queue.get()


@pytest.mark.parametrize("method", PIXEL_METHODS)
def test_dmc_cartpole_pixels(method: str):
    result_queue = multiprocessing.Queue()
    # Create a multiprocessing.Process and pass the result queue as an argument
    process = multiprocessing.Process(
        target=run_cmd,
        args=(
            COMMON_HYDRA_OVERRIDES
            + [
                f"method={method}",
                "env=dmc/cartpole_balance",
                "pixels=true",
                "frame_stack=3",
                "method.use_augmentation=true",
                "replay.size=50000",
            ],
            800,
            result_queue,
        ),
    )
    # Start the process
    process.start()
    # Wait for the process to finish
    process.join()
    # Check if the process terminated successfully (exit code 0) or not
    if process.exitcode != 0:
        raise result_queue.get()


# noqa: E501 TODO: reach target is not stable enough at the moment to do integration test with rlbench
# @pytest.mark.parametrize("method", METHODS)
# def test_rlbench_reach_target_no_pixels(method: str):
#     run_cmd(
#         COMMON_HYDRA_OVERRIDES
#         + [
#             f"method={method}",
#             "env=rlbench/reach_target",
#             "pixels=false",
#             "action_sequence=4",
#         ],
#         0.8,
#     )


# @pytest.mark.parametrize("method", METHODS)
# def test_rlbench_reach_target_pixels(method: str):
# noqa: E501 run_cmd(COMMON_HYDRA_OVERRIDES + [f"method={method}", "env=rlbench/reach_target", "env.cameras=[wrist, front]", "pixels=true"], 0.8)

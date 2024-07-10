"""Tests model-based rl methods."""
import multiprocessing
import tempfile

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from tests.unit.method.base import Base, train_and_shutdown


DEFAULT_OVERRIDES = [
    "num_train_frames=100",
    "num_gpus=0",
    "num_eval_episodes=2",
    "replay_size_before_train=5",
    "batch_size=2",
    "replay.sequential=true",
    "replay.size=20",
    "replay.num_workers=0",
    "replay.pin_memory=false",
    "env.episode_length=5",
    "method.use_torch_compile=false",
    "method.use_amp=false",
    "visual_observation_shape=[64,64]",
    "frame_stack=1",
    "frame_stack_on_channel=false",
    "replay.transition_seq_len=2",
]

LIGHT_RSSM_OVERRIDES = [
    "method.dynamics_model.stoch=4",
    "method.dynamics_model.discrete=4",
    "method.dynamics_model.deter=16",
    "method.dynamics_model.hidden=16",
]

MWM_OVERRIDES = [
    "method.mae_pretrain_steps=0",
    "method.mae_warmup=0",
]


@pytest.mark.parametrize(
    "method,cfg_params",
    [
        ("dreamerv3", DEFAULT_OVERRIDES + LIGHT_RSSM_OVERRIDES),
        ("mwm", DEFAULT_OVERRIDES + LIGHT_RSSM_OVERRIDES + MWM_OVERRIDES),
    ],
)
class TestModelBasedRLMethods(Base):
    def test_rlbench_without_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        if "mwm" in method:
            pytest.skip("MWM does not support state-only environments.")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + ["pixels=false", "env=rlbench/reach_target"],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            # RLBench needs to be run with multiprocess
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_rlbench_with_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method + cfg_params + ["pixels=true", "env=rlbench/reach_target"],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            # RLBench needs to be run with multiprocess
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_dmc_without_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        if "mwm" in method:
            pytest.skip("MWM does not support state-only environments.")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=false",
                "env=dmc/acrobot_swingup",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_dmc_with_pixels_with_action_many_features(self, method, cfg_params):
        # TODO: We should find a better way to test new features.
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=true",
                "env=dmc/acrobot_swingup",
                "action_sequence=1",
                "method.num_critics=4",
                "use_standardization=true",
                "use_min_max_normalization=true",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_dmc_with_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=true",
                "env=dmc/acrobot_swingup",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_bigym_without_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        if "mwm" in method:
            pytest.skip("MWM does not support state-only environments.")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method + cfg_params + ["pixels=false", "env=bigym/reach_target"],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_bigym_with_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method + cfg_params + ["pixels=true", "env=bigym/reach_target"],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

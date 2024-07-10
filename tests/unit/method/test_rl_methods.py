"""Tests rl methods."""
import multiprocessing
import tempfile

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from tests.unit.method.base import Base, train_and_shutdown


DEFAULT_OVERRIDES = [
    "num_train_frames=100",
    "num_gpus=0",
    "num_eval_episodes=2",
    "replay_size_before_train=5",
    "batch_size=2",
    "replay.size=20",
    "replay.num_workers=0",
    "replay.pin_memory=false",
    "env.episode_length=5",
]

DISTRIBUTIONAL_METHODS_OVERRIDES = [
    "method.distributional_critic=true",
    "method.critic_model.output_shape=[251, 1]",
]


@pytest.mark.parametrize(
    "method,cfg_params",
    [
        ("drqv2", DEFAULT_OVERRIDES),
        ("drm", DEFAULT_OVERRIDES),
        ("alix", DEFAULT_OVERRIDES),
        ("sac_lix", DEFAULT_OVERRIDES),
        ("iql_drqv2", DEFAULT_OVERRIDES),
        ("cqn", DEFAULT_OVERRIDES),
        ("drqv2", DEFAULT_OVERRIDES + DISTRIBUTIONAL_METHODS_OVERRIDES),
        ("drm", DEFAULT_OVERRIDES + DISTRIBUTIONAL_METHODS_OVERRIDES),
        ("alix", DEFAULT_OVERRIDES + DISTRIBUTIONAL_METHODS_OVERRIDES),
        ("sac_lix", DEFAULT_OVERRIDES + DISTRIBUTIONAL_METHODS_OVERRIDES),
    ],
)
class TestRLMethods(Base):
    def test_rlbench_without_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
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

    @pytest.mark.parametrize(
        ["img_shape", "encoder_model", "view_fusion_model"],
        [
            (
                (84, 84),
                {
                    "_target_": "robobase.models.EncoderCNNMultiViewDownsampleWithStrides",
                    "_partial_": True,
                    "num_downsample_convs": 1,
                    "num_post_downsample_convs": 3,
                },
                {
                    "_target_": "robobase.models.FusionMultiCamFeature",
                    "_partial_": True,
                },
            ),
            # TODO: Can't seem to get the below network to work on Bamboo.
            # (
            #     (224, 224),
            #     {
            #         "_target_": "robobase.models.EncoderMVPMultiView",
            #         "_partial_": True,
            #         "name": "vits-mae-hoi",
            #     },
            #     {
            #         "_target_": "robobase.models.FusionMultiCamFeature",
            #         "_partial_": True,
            #     },
            # ),
        ],
    )
    def test_rlbench_with_pixels(
        self, method, cfg_params, img_shape, encoder_model, view_fusion_model
    ):
        if "lix" in method:
            pytest.skip("Lix-based methods requires special networks.")
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method + cfg_params + ["pixels=true", "env=rlbench/reach_target"],
        )
        cfg.visual_observation_shape = img_shape
        cfg.method.encoder_model = OmegaConf.create(encoder_model)
        cfg.method.view_fusion_model = OmegaConf.create(view_fusion_model)
        with tempfile.TemporaryDirectory() as tempdir:
            # RLBench needs to be run with multiprocess
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_dmc_without_pixels(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=false",
                "env=dmc/acrobot_swingup",
                "frame_stack=2",
                "frame_stack_on_channel=true",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    @pytest.mark.parametrize(
        "intrinsic,",
        ["rnd", "icm"],
    )
    def test_dmc_without_pixels_with_intrinsic(self, method, cfg_params, intrinsic):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=false",
                "env=dmc/acrobot_swingup",
                "frame_stack=2",
                "frame_stack_on_channel=true",
                f"intrinsic_reward_module={intrinsic}",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    @pytest.mark.parametrize(
        "intrinsic,",
        ["rnd", "icm"],
    )
    def test_dmc_with_pixels_with_intrinsic(self, method, cfg_params, intrinsic):
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
                "frame_stack=2",
                "frame_stack_on_channel=true",
                f"intrinsic_reward_module={intrinsic}",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_dmc_without_pixels_with_action_many_features(self, method, cfg_params):
        # TODO: We should find a better way to test new features.
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        if "cqn" in method:
            pytest.skip("CQN does not support num_critics.")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=false",
                "env=dmc/acrobot_swingup",
                "frame_stack=2",
                "frame_stack_on_channel=true",
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

    def test_dmc_without_pixels_stack_frames_rnn(self, method, cfg_params):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=false",
                "env=dmc/acrobot_swingup",
                "frame_stack=2",
                "frame_stack_on_channel=false",
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
                "frame_stack=2",
                "frame_stack_on_channel=true",
            ],
        )
        with tempfile.TemporaryDirectory() as tempdir:
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_dmc_with_pixels_stack_frames_rnn(self, method, cfg_params):
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
                "frame_stack=2",
                "frame_stack_on_channel=false",
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

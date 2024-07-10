"""Tests IL algos."""
import multiprocessing
import tempfile

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from omegaconf import OmegaConf

from tests.unit.method.base import Base, train_and_shutdown


COMMON_OVERRIDES = [
    "num_pretrain_steps=2",
    "num_train_frames=2",
    "num_gpus=0",
    "num_eval_episodes=2",
    "replay_size_before_train=10",
    "batch_size=2",
    "replay.size=500",
    "replay.nstep=1",
    "replay.num_workers=0",
    "replay.pin_memory=false",
    "log_pretrain_every=1",
    "env.episode_length=10",
]

DEFAULT_OVERRIDES = COMMON_OVERRIDES + ["action_sequence=4"]
DIFFUSION_OVERRIDES = COMMON_OVERRIDES + [
    "method.num_diffusion_iters=2",
    "action_sequence=20",
]


@pytest.mark.parametrize(
    "method,cfg_params",
    [
        ("bc", DEFAULT_OVERRIDES),
        ("diffusion", DIFFUSION_OVERRIDES),
        ("act", DEFAULT_OVERRIDES),
    ],
)
class TestILMethods(Base):
    def test_rlbench_without_pixels(self, method, cfg_params):
        if method == "act":
            pytest.skip("ACT does not support state-only environments.")
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method
            + cfg_params
            + [
                "pixels=false",
                "env=rlbench/reach_target",
                "env.action_mode=JOINT_POSITION",
                "demos=1",
            ],
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
        ],
    )
    def test_rlbench_with_pixels(
        self, method, cfg_params, img_shape, encoder_model, view_fusion_model
    ):
        GlobalHydra.instance().clear()
        initialize(config_path="../../../robobase/cfgs")
        method_cfg = ["method=" + method]
        cfg = compose(
            config_name="robobase_config",
            overrides=method_cfg
            + cfg_params
            + [
                "pixels=true",
                "env=rlbench/reach_target",
                "env.action_mode=JOINT_POSITION",
                "demos=1",
            ],
        )

        if method == "act":
            encoder_model = {
                "_target_": "robobase.method.act.ImageEncoderACT",
                "_partial_": True,
            }
            actor_model = {
                "_target_": "robobase.models.multi_view_transformer"
                ".MultiViewTransformerEncoderDecoderACT",
                "_partial_": True,
                "num_queries": cfg.action_sequence,
            }
            cfg.visual_observation_shape = img_shape
            cfg.method.encoder_model = OmegaConf.create(encoder_model)
            cfg.method.actor_model = OmegaConf.create(actor_model)
        else:
            cfg.visual_observation_shape = img_shape
            cfg.method.encoder_model = OmegaConf.create(encoder_model)
            cfg.method.view_fusion_model = OmegaConf.create(view_fusion_model)

        with tempfile.TemporaryDirectory() as tempdir:
            # RLBench needs to be run with multiprocess
            p = multiprocessing.Process(target=train_and_shutdown, args=(cfg, tempdir))
            p.start()
            p.join()
            assert not p.exitcode

    def test_save_load_snapshot(self, method, cfg_params):
        new_params = cfg_params + [
            "env=rlbench/reach_target",
            "env.action_mode=JOINT_POSITION",
            "demos=1",
        ]
        super().test_save_load_snapshot(method, new_params)

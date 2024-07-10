import numpy as np
from robobase.method.utils import extract_many_from_spec

from robobase.models.lix_utils import analysis_optimizers
from robobase.method.drqv2 import DrQV2
from robobase.models.lix_utils.analysis_modules import LIXModule


class ALIX(DrQV2):
    """Implementation of Adaptive Local SIgnal MiXing (A-LIX).

    Cetin et al. Stabilizing Off-Policy Deep Reinforcement Learning from Pixels
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_augmentation"] = False
        super().__init__(*args, **kwargs)

    def build_encoder(self):
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            rgb_shapes = [s.shape for s in rgb_spaces.values()]
            assert np.all(
                [sh == rgb_shapes[0] for sh in rgb_shapes]
            ), "Expected all RGB obs to be same shape."

            num_views = len(rgb_shapes)
            if self.frame_stack_on_channel:
                obs_shape = (np.prod(rgb_shapes[0][:2]), *rgb_shapes[0][2:])
            else:
                # T is folded into batch
                obs_shape = rgb_shapes[0][1:]
            self.encoder = self.encoder_model(input_shape=(num_views, *obs_shape))
            if not isinstance(self.encoder, LIXModule):
                raise ValueError("Encoder must be of type LIXModule.")
            self.encoder.to(self.device)
            self.encoder_opt = (
                analysis_optimizers.custom_parameterized_aug_optimizer_builder(
                    encoder_lr=self.encoder_lr, lr=2e-3, betas=[0.5, 0.999]
                )(self.encoder)
            )

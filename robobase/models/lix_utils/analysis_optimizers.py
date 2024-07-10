import torch as th

from robobase.models.lix_utils.analysis_layers import ParameterizedReg


def custom_parameterized_aug_optimizer_builder(encoder_lr, **kwargs):
    """Apply different optimizer parameters for S"""

    def make_optimizer(
        encoder,
    ):
        assert isinstance(encoder.aug, ParameterizedReg)
        encoder_params = list(encoder.parameters())
        encoder_aug_parameters = list(encoder.aug.parameters())
        encoder_non_aug_parameters = [
            p
            for p in encoder_params
            if all([p is not aug_p for aug_p in encoder_aug_parameters])
        ]
        return th.optim.Adam(
            [
                {"params": encoder_non_aug_parameters},
                {"params": encoder_aug_parameters, **kwargs},
            ],
            lr=encoder_lr,
        )

    return make_optimizer

from abc import ABC
from typing import Tuple

import numpy as np
import torch.nn as nn

from robobase.models.lix_utils.analysis_layers import (
    NonLearnableParameterizedRegWrapper,
)
from robobase.models.lix_utils import analysis_layers
from robobase.models import EncoderCNNMultiViewDownsampleWithStrides
from robobase.utils import weight_init


class LIXModule(ABC):
    pass


class EncoderAllFeatTiedRegularizedCNNMultiViewDownsampleWithStrides(
    EncoderCNNMultiViewDownsampleWithStrides, LIXModule
):
    """Encoder with the same regularization applied after every layer, and with the
    regularization parameter tuned only with the final layer's feature gradients."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        num_downsample_convs: int,
        num_post_downsample_convs: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__(
            input_shape,
            num_downsample_convs,
            num_post_downsample_convs,
            channels,
            kernel_size,
        )
        assert (
            len(input_shape) == 4
        ), f"Expected shape (V, C, H, W), but got {input_shape}"
        self._input_shape = input_shape
        num_cameras = input_shape[0]
        self.convs_per_cam = nn.ModuleList()
        final_channels = 0
        self.aug = analysis_layers.ParameterizedReg(
            analysis_layers.LocalSignalMixing(2, True),
            0.5,
            param_grad_fn="alix_param_grad",
            param_grad_fn_args=[3, 0.535, 1e-20],
        )
        for _ in range(num_cameras):
            resolution = np.array(input_shape[2:])
            net = []
            ch = input_shape[1]
            for _ in range(num_downsample_convs):
                net.append(
                    nn.Conv2d(
                        ch,
                        channels,
                        kernel_size=kernel_size,
                        stride=2,
                    )
                )
                net.append(nn.ReLU())
                net.append(NonLearnableParameterizedRegWrapper(self.aug))
                ch = channels
                resolution = np.ceil(resolution / 2) - (kernel_size // 2)
            for _ in range(num_post_downsample_convs):
                net.append(
                    nn.Conv2d(
                        ch,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                    )
                )
                net.append(nn.ReLU())
                net.append(NonLearnableParameterizedRegWrapper(self.aug))
                ch = channels
                resolution -= (kernel_size // 2) * 2
            net.append(self.aug)
            self.convs_per_cam.append(nn.Sequential(*net))
            final_channels = int(channels * resolution.prod())
        self._output_shape = (num_cameras, final_channels)
        self.apply(weight_init)

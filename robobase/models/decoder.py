from abc import ABC
import math
from typing import Tuple

import numpy as np
import torch
from torch import nn as nn

from robobase import utils
from robobase.models.core import (
    RoboBaseModule,
    get_activation_fn_from_str,
    get_normalization_fn_from_str,
)


class DecoderModule(RoboBaseModule, ABC):
    def __init__(self, input_shape: tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        assert len(input_shape) == 1, f"Expected input shape (C), but got {input_shape}"


class DecoderCNNMultiView(DecoderModule):
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int, int, int, int],
        min_res: 4,
        channels: int = 32,
        kernel_size: int = 4,
        channels_multiplier: int = 1,
        activation: str = "relu",
        norm: str = "identity",
    ):
        super().__init__(input_shape)
        num_cameras = output_shape[0]
        num_layers = int(np.log2(output_shape[2]) - np.log2(min_res))
        assert output_shape[2] == output_shape[3], "Only support square images"
        assert (
            output_shape[2] > 0 and (output_shape[2] & (output_shape[2] - 1)) == 0
        ), f"{output_shape[2]} is not a power of 2"
        linear_output_channels = (
            (min_res**2) * channels * (channels_multiplier ** (num_layers - 1))
        )
        self.activation_fn = get_activation_fn_from_str(activation)
        self.norm_fn = get_normalization_fn_from_str(norm)
        self.linears_per_cam = nn.ModuleList()
        self.convs_per_cam = nn.ModuleList()

        for i in range(num_cameras):
            self.linears_per_cam.append(
                nn.Linear(input_shape[0], linear_output_channels)
            )
            net = []
            input_channels = (linear_output_channels) // (min_res**2)
            for i in range(num_layers):
                is_last_layer = i == num_layers - 1
                if is_last_layer:
                    output_channels = output_shape[1]
                else:
                    output_channels = input_channels // channels_multiplier
                padding, output_padding = self.calculate_same_pad(kernel_size, 2, 1)
                net.append(
                    nn.ConvTranspose2d(
                        input_channels,
                        output_channels,
                        kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False if not is_last_layer else True,
                    )
                )
                if not is_last_layer:
                    net.append(self.norm_fn(output_channels))
                    net.append(self.activation_fn())
                input_channels = output_channels
            self.convs_per_cam.append(nn.Sequential(*net))
        self._min_res = min_res
        self._output_shape = output_shape
        self.apply(utils.weight_init)

    @property
    def output_shape(self):
        return self._output_shape

    def calculate_same_pad(self, kernel, stride, dilation):
        val = dilation * (kernel - 1) - stride + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def initialize_output_layer(self, initialize_fn):
        """Initialize output layer with specified initialize function

        Could be useful when a user wants to specify the initialization scheme for
        the output layer (e.g., zero initialization)
        """
        for conv_per_cam in self.convs_per_cam:
            conv_per_cam[-1].apply(initialize_fn)

    def forward(self, features):
        assert (
            self.input_shape == features.shape[1:]
        ), f"expected input shape {self.input_shape} but got {features.shape[1:]}"

        num_cameras = self.output_shape[0]
        outs = []
        for i in range(num_cameras):
            x = self.linears_per_cam[i](features)
            x = x.reshape(
                [-1, x.shape[-1] // (self._min_res**2), self._min_res, self._min_res]
            )
            x = self.convs_per_cam[i](x)
            outs.append(x)
        fused = torch.stack(outs, 1)
        assert (
            fused.shape[1:] == self.output_shape
        ), f"Expected output {self.output_shape}, but got {fused.shape[1:]}"
        return fused

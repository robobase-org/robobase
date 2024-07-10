from abc import ABC

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


class FullyConnectedModule(RoboBaseModule, ABC):
    def __init__(self, input_shapes: dict[str, tuple], output_shape: int | Tuple):
        super().__init__()
        self.input_shapes = input_shapes
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        # Hydra cannot take tuple inputs, this has to be done
        self._output_shape = tuple(output_shape)
        self.eval_env_running = False

    @property
    def output_shape(self):
        return self._output_shape

    def reset(self, env_index: int):
        pass


class RNNFullyConnectedModule(FullyConnectedModule, ABC):
    def __init__(
        self,
        input_shapes: dict[str, tuple],
        output_shape: int,
        num_envs: int,  # train and eval
        num_rnn_layers: int,
        rnn_hidden_size: int,
    ):
        """Base class for all rnn-based fully connected modules.

        This module makes the assumption that input shapes are either 1-dim (Z,), or
        2-dim (T, Z), where T is the frame stack.
        That means, you will need to implement a use case for both.

        Args:
            input_shapes: Dictionary specifying input shapes for different keys.
            output_shape: Size of the output layer.
            num_envs: The total number of envs (train and eval).
                Used for tracking hidden state.
            num_rnn_layers: Number of rnn layers used.
            rnn_hidden_size: Size of the hidden state in the rnn.
        """
        super().__init__(input_shapes, output_shape)
        self.input_shapes = input_shapes
        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.dim1_keys = []
        self.dim2_keys = []
        self._assert_shapes()

        self.input_time_dim_size = -1
        for v in input_shapes.values():
            if len(v) == 2:
                self.input_time_dim_size = v[0]  # T part of tuple
                break
        hidden_state = torch.zeros(
            (self.num_rnn_layers, self.num_envs, self.rnn_hidden_size),
            requires_grad=False,
        )
        self.register_buffer("hidden_state", hidden_state)

    def _assert_shapes(self):
        for v in self.input_shapes.values():
            if len(v) not in [1, 2]:
                raise ValueError(
                    "FullyConnectedModule only supports inputs "
                    "of shape (T, Z), or (Z,)."
                )

    def reset(self, env_index: int):
        self.hidden_state[:, env_index] *= 0


class MLPWithBottleneckFeatures(RNNFullyConnectedModule):
    def __init__(
        self,
        keys_to_bottleneck: list[str],
        bottleneck_size: int,
        norm_after_bottleneck: bool,
        tanh_after_bottleneck: bool,
        mlp_nodes: list[int],
        activation: str = "relu",
        norm: str = "identity",
        linear_bias: bool = True,
        *args,
        **kwargs,
    ):
        """Creates a multi-layer perceptron (MLP) with a bottleneck for inputs.

        This model constructs an MLP with the ability to introduce bottleneck input
        layers for specific keys in the input dictionary. E.g. we may want to compress
        image features but not low dimensional input.

        The bottleneck can be of the following style (customised through the flags):
        (a) nn.Tanh(nn.LayerNorm(nn.Linear(in)))
        (b) nn.LayerNorm(nn.Linear(in))
        (c) nn.Tanh(nn.Linear(in))
        (d) nn.Linear(in)

        Args:
            keys_to_bottleneck: Keys from input_shapes to apply the bottleneck feature.
            bottleneck_size: Size of the bottleneck layer.
            norm_after_bottleneck: Whether to apply normalization after bottleneck.
            tanh_after_bottleneck: Whether to apply Tanh activation after bottleneck.
            mlp_nodes: List specifying the number of nodes in each hidden layer.
            num_rnn_layers: Number of rnn layers when not using
                channel-based frame-stacking.
            rnn_hidden_size: Size of the hidden state in the rnn.
            activation: Activation function str to be used in the hidden layers.
            norm: Normalization class str to be used in the hidden layers.
            linear_bias: Whether to set bias=True in nn.Linear
        """
        super().__init__(*args, **kwargs)
        self.keys_to_bottleneck = keys_to_bottleneck
        self.bottleneck_size = bottleneck_size
        self.mlp_nodes = mlp_nodes
        self.activation_fn = get_activation_fn_from_str(activation)
        self.norm_fn = get_normalization_fn_from_str(norm)
        input_preprocess_modules = {}
        for k in keys_to_bottleneck:
            if k not in self.input_shapes:
                continue
            net = [
                nn.Linear(self.input_shapes[k][-1], bottleneck_size, bias=linear_bias)
            ]
            if norm_after_bottleneck:
                net.append(nn.LayerNorm(bottleneck_size))
            if tanh_after_bottleneck:
                net.append(nn.Tanh())
            input_preprocess_modules[k] = nn.Sequential(*net)
        self.input_preprocess_modules = nn.ModuleDict(input_preprocess_modules)

        inputs = [
            v[-1]
            for k, v in self.input_shapes.items()
            if k not in self.keys_to_bottleneck
        ]
        inputs_for_bottleneck = [
            bottleneck_size
            for k, v in self.input_shapes.items()
            if k in self.keys_to_bottleneck
        ]
        in_size = np.sum(inputs + inputs_for_bottleneck)
        self.time_dim_rnn = None
        if self.input_time_dim_size > 0:
            self.time_dim_rnn = nn.GRU(
                in_size, self.rnn_hidden_size, self.num_rnn_layers, batch_first=True
            )
            in_size = self.rnn_hidden_size
        main_mlp = []
        for nodes in mlp_nodes:
            main_mlp.append(nn.Linear(in_size, nodes, bias=linear_bias))
            main_mlp.append(self.norm_fn(nodes))
            main_mlp.append(self.activation_fn())
            in_size = nodes
        self.main_mlp = nn.Sequential(*main_mlp)

        out_mlp = []
        out_mlp.append(nn.Linear(in_size, np.prod(self._output_shape)))
        out_mlp.append(Reshape(self._output_shape, 1))
        self.out_mlp = nn.Sequential(*out_mlp)

        self.apply(utils.weight_init)
        self._need_flatten = True

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(x, dict), "Expected input to be dict of str to tensors."
        if self.training and self.time_dim_rnn is not None and self._need_flatten:
            self._need_flatten = False
            self.time_dim_rnn.flatten_parameters()
        if not self.training and not self._need_flatten:
            # We have been in eval mode and need to flatten next time ee train
            self._need_flatten = True
        inputs = [v for k, v in x.items() if k not in self.keys_to_bottleneck]
        feats = []
        for k, mlp in self.input_preprocess_modules.items():
            feats.append(mlp(x[k]))
        feats.extend(inputs)
        expanded_feats = []
        for _x in feats:
            if self.input_time_dim_size > 0 and _x.ndim == 2:  # i.e. (B, Z)
                # Create a time dim for this tuple
                _x = _x.unsqueeze(1).repeat(1, self.input_time_dim_size, 1)
            expanded_feats.append(_x)
        feats = torch.cat(expanded_feats, dim=-1)
        if self.input_time_dim_size > 0:
            h = None
            if not self.training:
                h = (
                    self.hidden_state[:, -1:]
                    if self.eval_env_running
                    else self.hidden_state[:, :-1]
                ).contiguous()
                feats = feats[:, -1:]  # get last feats
            feats, h = self.time_dim_rnn(feats, h)
            feats = feats[:, -1]  # get last feats
            if not self.training:
                if self.eval_env_running:
                    self.hidden_state[:, -1:] = h.detach()
                else:
                    self.hidden_state[:, :-1] = h.detach()
        return self.out_mlp(self.main_mlp(feats))

    def initialize_output_layer(self, initialize_fn):
        """Initialize output layer with specified initialize function

        Could be useful when a user wants to specify the initialization scheme for
        the output layer (e.g., zero initialization)
        """
        output_layer = self.out_mlp[-2]
        assert isinstance(output_layer, nn.Linear)
        output_layer.apply(initialize_fn)


class MLPWithBottleneckFeaturesAndWithoutHead(MLPWithBottleneckFeatures):
    def __init__(self, *args, **kwargs):
        """Creates MLP with a bottleneck for inputs, but without head layer"""
        super().__init__(*args, **kwargs)
        del self.out_mlp
        self.out_mlp = nn.Identity()

    @property
    def output_shape(self):
        return (self.mlp_nodes[-1],)

    def initialize_output_layer(self, initialize_fn):
        raise ValueError("This class does not support initialize_output_layer")


class GRU(nn.Module):
    def __init__(self, nodes: int, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self._rnn = nn.GRU(nodes, nodes, sequence_length, batch_first=True)
        self._hidden = nn.Parameter(
            data=torch.zeros(self.sequence_length, 1, nodes),
            requires_grad=False,
        )

    def forward(self, x):
        bs = x.shape[0]
        out, _ = self._rnn(
            x.unsqueeze(1).repeat((1, self.sequence_length, 1)),
            self._hidden.repeat(1, bs, 1),
        )
        return out


class Reshape(nn.Module):
    def __init__(self, shapes, last_dim_to_keep=1):
        super(Reshape, self).__init__()
        self.shapes = shapes
        assert last_dim_to_keep != 0
        self.last_dim_to_keep = last_dim_to_keep

    def forward(self, x):
        return x.view(x.shape[: -self.last_dim_to_keep] + self.shapes)

    def __repr__(self):
        return (
            f"Reshape(shapes={self.shapes}, last_dim_to_keep={self.last_dim_to_keep})"
        )


class MLPWithBottleneckFeaturesAndSequenceOutput(MLPWithBottleneckFeatures):
    def __init__(
        self,
        output_sequence_network_type: str,
        output_sequence_length: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.output_sequence_network_type = output_sequence_network_type.lower()
        self.output_sequence_length = output_sequence_length
        if self.output_sequence_network_type not in ["rnn", "mlp"]:
            raise ValueError(
                f"output_sequence_network_type: {self.output_sequence_network_type} "
                "not supported."
            )

        del self.out_mlp  # We will overwrite this.
        in_size = self.mlp_nodes[-1]
        out_mlp = []
        if output_sequence_length > 1 and self.output_sequence_network_type != "mlp":
            out_mlp.extend(
                [
                    GRU(in_size, output_sequence_length),
                    nn.Linear(in_size, np.prod(self._output_shape)),
                    Reshape(self._output_shape, 1),
                ]
            )
        else:
            out_mlp.extend(
                [
                    nn.Linear(
                        in_size,
                        self.output_sequence_length * np.prod(self._output_shape),
                    ),
                    Reshape((self.output_sequence_length, *self._output_shape), 1),
                ]
            )
        self.out_mlp = nn.Sequential(*out_mlp)
        self.out_mlp.apply(utils.weight_init)

    @property
    def output_shape(self):
        return self.output_sequence_length, *self._output_shape

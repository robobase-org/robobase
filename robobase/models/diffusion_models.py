"""
Code taken from: https://github.com/real-stanford/diffusion_policy/tree/main
"""

from functools import partial
from typing import Tuple, Callable
import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np

from robobase.models.fully_connected import (
    FullyConnectedModule,
    RNNFullyConnectedModule,
    Reshape,
)
from robobase.models.core import (
    get_activation_fn_from_str,
    get_normalization_fn_from_str,
)
from robobase import utils


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class MLPWithBottleneckFeaturesForDiffusion(RNNFullyConnectedModule):
    def __init__(
        self,
        diffusion_step_embed_dim: int,
        sequence_length: int,
        keys_to_bottleneck: list[str],
        bottleneck_size: int,
        norm_after_bottleneck: bool,
        tanh_after_bottleneck: bool,
        mlp_nodes: list[int],
        activation: str = "relu",
        norm: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        self.keys_to_bottleneck = keys_to_bottleneck
        self.bottleneck_size = bottleneck_size
        self.sequence_length = sequence_length
        self.mlp_nodes = mlp_nodes
        self.activation_fn = get_activation_fn_from_str(activation)
        self.norm_fn = get_normalization_fn_from_str(norm)
        input_preprocess_modules = {}
        for k in keys_to_bottleneck:
            if k not in self.input_shapes:
                continue
            net = [nn.Linear(self.input_shapes[k][-1], bottleneck_size)]
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
        in_size = np.sum(inputs + inputs_for_bottleneck) + dsed  # add encoding size
        if self.input_time_dim_size > 0:
            self.time_dim_rnn = nn.GRU(
                in_size, self.rnn_hidden_size, self.num_rnn_layers, batch_first=True
            )
            in_size = self.rnn_hidden_size
        main_mlp = []
        for nodes in mlp_nodes:
            main_mlp.append(nn.Linear(in_size, nodes))
            main_mlp.append(self.norm_fn(nodes))
            main_mlp.append(self.activation_fn())
            in_size = nodes
        self.main_mlp = nn.Sequential(*main_mlp)

        out_mlp = []
        out_mlp.append(nn.Linear(in_size, np.prod(self._output_shape)))
        out_mlp.append(Reshape(self._output_shape, 1))
        self.out_mlp = nn.Sequential(*out_mlp)

        self.apply(utils.weight_init)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: (B,T,input_dim)
                timestep: (B,) or int, diffusion step
                local_cond: (B,T,local_cond_dim)
                global_cond: (B,global_cond_dim)
                output: (B,T,input_dim)
        """
        assert isinstance(x, dict), "Expected input to be dict of str to tensors."
        action = x["actions"]
        timestep = x["timestep"]
        features = x.get("features", None)
        assert action.ndim == 3
        assert features is None or features.ndim == 2

        # preprocess action
        # (B,T,C) -> (B,C,T)
        action = action.moveaxis(-1, -2)
        assert action.shape[-1] == 1
        b, c, t = action.shape[0], action.shape[1], action.shape[2]
        action = action.view(b, c * t)

        # preprocess timestep, encode it
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU.
            # So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=action.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(action.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(action.shape[0])
        timestep_embedding = self.diffusion_step_encoder(timesteps)

        # create new x to pass through MLP
        x = {"actions": action, "timestep": timestep_embedding, "features": features}

        # MLP with bottleneck
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
        out = self.out_mlp(self.main_mlp(feats))
        out = out.view(b, c, t)
        out = out.moveaxis(-1, -2)
        return out

    def initialize_output_layer(self, initialize_fn):
        """Initialize output layer with specified initialize function

        Could be useful when a user wants to specify the initialization scheme for
        the output layer (e.g., zero initialization)
        """
        output_layer = self.out_mlp[-2]
        assert isinstance(output_layer, nn.Linear)
        output_layer.apply(initialize_fn)

    @property
    def output_shape(self):
        # TODO do we really need to do this?????
        return (self.sequence_length, *self._output_shape)


class ConditionalUnet1D(FullyConnectedModule):
    def __init__(
        self,
        input_shapes: dict[str, tuple],
        output_shape: int,
        sequence_length: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__(input_shapes, output_shape)
        self.sequence_length = sequence_length
        if sequence_length < 4:
            raise ValueError("sequence_length must be >= 4.")
        if "actions" not in input_shapes:
            raise ValueError("This model requires there to be an action input.")
        input_dim = input_shapes["actions"][-1]
        features_dim = input_shapes.get("features", None)[-1]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if features_dim is not None:
            cond_dim += features_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    @property
    def preferred_optimiser(self) -> callable:
        return partial(torch.optim.AdamW, params=self.parameters(), weight_decay=1e-6)

    @property
    def output_shape(self):
        return (self.sequence_length, *self._output_shape)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        assert isinstance(x, dict), "Expected input to be dict of str to tensors."
        sample = x["actions"]
        timestep = x["timestep"]
        features_dim = x.get("features", None)
        assert sample.ndim == 3
        assert features_dim is None or features_dim.ndim == 2

        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU.
            # So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if features_dim is not None:
            global_feature = torch.cat([global_feature, features_dim], axis=-1)

        # encode local features
        h_local = list()

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class TransformerForDiffusion(FullyConnectedModule):
    def __init__(
        self,
        input_shapes: dict[str, tuple],
        output_shape: int,
        sequence_length: int,
        n_obs_steps: int = None,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        n_cond_layers: int = 4,
    ) -> None:
        super().__init__(input_shapes, output_shape)
        self.sequence_length = sequence_length
        input_dim = input_shapes["actions"][-1]
        cond_dim = input_shapes["features"][-1]

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = sequence_length

        T = sequence_length
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim is not None
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied
            # to the left in the input sequence
            # torch.nn.Transformer uses additive mask
            # as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf
            # and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
                mask = t >= (
                    s - 1
                )  # add one dimension since time is the first token in cond
                mask = (
                    mask.float()
                    .masked_fill(mask == 0, float("-inf"))
                    .masked_fill(mask == 1, float(0.0))
                )
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_shape)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    @property
    def preferred_optimiser(self) -> callable:
        return partial(
            self.noise_pred_net.configure_optimizers,
            weight_decay=1e-3,
            betas=(0.9, 0.95),
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple
        and is being very defensive:
        We are separating out all parameters of the model into two buckets:
        those that will experience weight decay for regularization and
        those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and
        # won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter
        # in the root GPT module as not decayed
        no_decay.add("pos_emb")
        # no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    @property
    def output_shape(self):
        return self.sequence_length, *self._output_shape

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        sample = x["actions"]
        timestep = x["timestep"]
        cond = x["features"]
        assert sample.ndim == 3
        assert cond is None or cond.ndim == 2
        if cond is not None:
            # Assume view/temporal fusion has been done elsewhere
            cond = cond.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU.
            # So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:, 1:, :]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)

            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask
            )
            # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

from abc import ABC
from functools import partial
from typing import Tuple

import numpy as np
import timm
import torch
from timm.models.vision_transformer import Block
from torch import nn as nn
from torch.nn import functional as F

from robobase import utils
from robobase.models.core import (
    RoboBaseModule,
    get_activation_fn_from_str,
    get_normalization_fn_from_str,
)
from robobase.models.utils import (
    MultiViewConvEmbed,
    MultiViewPatchEmbed,
    get_2d_sincos_pos_embed,
)


class EncoderModule(RoboBaseModule, ABC):
    def __init__(self, input_shape: tuple[int, int, int, int]):
        super().__init__()
        self.input_shape = input_shape
        assert (
            len(input_shape) == 4
        ), f"Expected shape (V, C, H, W), but got {input_shape}"


class EncoderCNNMultiViewDownsampleWithStrides(EncoderModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        num_downsample_convs: int = 1,
        num_post_downsample_convs: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        padding: int = 0,
        channels_multiplier: int = 1,
        activation: str = "relu",
        norm: str = "identity",
        normalise_inputs: bool = True,
    ):
        super().__init__(input_shape)
        self._normalise_inputs = normalise_inputs
        num_cameras = input_shape[0]
        self.activation_fn = get_activation_fn_from_str(activation)
        self.norm_fn = get_normalization_fn_from_str(norm)
        self.convs_per_cam = nn.ModuleList()
        final_channels = 0
        for i in range(num_cameras):
            resolution = np.array(input_shape[2:])
            net = []
            input_channels = input_shape[1]
            output_channels = channels
            for _ in range(num_downsample_convs):
                net.append(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    )
                )
                net.append(self.norm_fn(output_channels))
                net.append(self.activation_fn())
                input_channels = output_channels
                output_channels *= channels_multiplier
                resolution = np.floor((resolution + 2 * padding - kernel_size) / 2) + 1
            for _ in range(num_post_downsample_convs):
                net.append(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    )
                )
                net.append(self.norm_fn(output_channels))
                net.append(self.activation_fn())
                input_channels = output_channels
                output_channels *= channels_multiplier
                resolution = np.floor((resolution + 2 * padding - kernel_size) / 1) + 1
            self.convs_per_cam.append(nn.Sequential(*net))
            final_channels = int(input_channels * resolution.prod())
        self._output_shape = (num_cameras, final_channels)
        self.apply(utils.weight_init)

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        if self._normalise_inputs:
            x = x / 255.0 - 0.5
        outs = []
        for _x, net in zip(x.unbind(1), self.convs_per_cam):
            outs.append(net(_x).view(-1, self.output_shape[-1]))
        fused = torch.stack(outs, 1)
        assert (
            fused.shape[1:] == self.output_shape
        ), f"Expected output {self.output_shape}, but got {fused.shape[1:]}"
        return fused


class EncoderMVPMultiView(EncoderModule):
    _OUT_DIM = {"vitb-mae-egosoup": 768, "vits-mae-hoi": 384}
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self, input_shape: Tuple[int, int, int, int], name: str = "vitb-mae-egosoup"
    ):
        super().__init__(input_shape)
        assert tuple(input_shape[2:]) == (
            224,
            224,
        ), f"MVP requires images of shape (224, 224), but got {input_shape[2:]}"
        assert input_shape[1] == 3, "MVP only supports channel of size 3"
        # Per-channel mean and standard deviation (in RGB order)
        try:
            import mvp
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
        except ImportError:
            raise ImportError(
                "Please run: pip install git+https://github.com/ir413/mvp"
            )
        self._mvp = mvp.load(name)
        self._mvp_out = EncoderMVPMultiView._OUT_DIM[name]
        self._mvp.freeze()

    def _color_norm(self, im, mean, std):
        """Performs per-channel normalization."""
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    def forward(self, x):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        x = self._color_norm(
            x / 255.0, EncoderMVPMultiView._MEAN, EncoderMVPMultiView._STD
        )
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                outs.append(self._mvp(_x).view(-1, self.output_shape[-1]))
            fused = torch.stack(outs, 1)
            assert (
                fused.shape[1:] == self.output_shape
            ), f"Expected output {self.output_shape}, but got {fused.shape[1:]}"
            return fused

    @property
    def output_shape(self):
        return self.input_shape[0], self._mvp_out


class ResNetEncoder(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        model: str,
    ):
        super().__init__(input_shape)
        assert input_shape[1] == 3, "ResNet only supports channel of size 3"
        self.model = timm.create_model(model, pretrained=True)
        self.model.eval()

    def forward(self, x: torch.Tensor):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        B, V = x.shape[:2]
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                _x = F.interpolate(
                    _x, size=(224, 224), mode="bilinear", align_corners=False
                )
                _x = self._color_norm(_x / 255.0)
                _x = self.model.forward_features(_x)
                _x = self.model.global_pool(_x)
                outs.append(_x.view(B, -1))
            fused = torch.stack(outs, 1)
        return fused

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = ResNetEncoder._MEAN
        std = ResNetEncoder._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    @property
    def output_shape(self):
        return (self.input_shape[0], self.model.num_features)


class DINOv2Encoder(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, input_shape: Tuple[int, int, int, int]):
        super().__init__(input_shape)
        self.input_shape = input_shape
        assert input_shape[1] == 3, "DINOv2 only supports channel of size 3"
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_lc"
        ).backbone
        self.model.eval()

    def forward(self, x: torch.Tensor):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        B, V = x.shape[:2]
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                _x = F.interpolate(
                    _x, size=(224, 224), mode="bilinear", align_corners=False
                )
                _x = self._color_norm(_x / 255.0)
                _x = self.model(_x)
                outs.append(_x.view(B, -1))
            fused = torch.stack(outs, 1)
        return fused

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = DINOv2Encoder._MEAN
        std = DINOv2Encoder._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    @property
    def output_shape(self):
        return (self.input_shape[0], self.model.embed_dim)


class R3MEncoder(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        model: str,
    ):
        super().__init__(input_shape)
        assert input_shape[1] == 3, "R3M only supports channel of size 3"

        try:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            from r3m import load_r3m

        except ImportError:
            raise ImportError(
                "Please run: pip install git+https://github.com/ir413/mvp"
            )

        if model == "r3m_resnet18":
            model = load_r3m("resnet18")
        elif model == "r3m_resnet34":
            model = load_r3m("resnet34")
        elif model == "r3m_resnet50":
            model = load_r3m("resnet50")
        else:
            raise ValueError(model)
        self.num_features = model.module.outdim
        self.model = model.module.convnet
        self.model.eval()

    def forward(self, x: torch.Tensor):
        assert (
            self.input_shape == x.shape[1:]
        ), f"expected input shape {self.input_shape} but got {x.shape[1:]}"
        B, V = x.shape[:2]
        with torch.no_grad():
            outs = []
            for _x in x.unbind(1):
                _x = F.interpolate(
                    _x, size=(224, 224), mode="bilinear", align_corners=False
                )
                _x = self._color_norm(_x / 255.0)
                _x = self.model(_x)
                outs.append(_x.view(B, -1))
            fused = torch.stack(outs, 1)
        return fused

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = ResNetEncoder._MEAN
        std = ResNetEncoder._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    @property
    def output_shape(self):
        return (self.input_shape[0], self.num_features)


class EncoderMultiViewVisionTransformer(EncoderModule):
    # Per-channel mean and standard deviation (in RGB order)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        patch_size: int,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 3,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        conv_embed: bool = True,
        reward_pred: bool = True,
    ):
        """
        This class is an implementation of Encoder class in
        Multi-View Masked Autoencoder (MV-MAE; https://arxiv.org/abs/2302.02408)

        Args:
            input_shape (Tuple[int, int, int, int]): V,C,H,W where V is the number of
            viewpoints and C is the channel of viewpoint.
            patch_size (int): Patch size. This should be the power of 2.
            embed_dim (int, optional): Embedding dimension of the encoder.
            depth (int, optional): Depth of the encoder.
            num_heads (int, optional): Number of heads in the encoder.
            decoder_embed_dim (int, optional): Embedding dimension of the decoder.
            decoder_depth (int, optional): Depth of the decoder.
            decoder_num_heads (int, optional): Number of heads in the decoder.
            mlp_ratio (float, optional): MLP ratio for linear layers in Transformer.
            norm_layer (nn.Module, optional): Type of layernorm in Transformer.
            conv_embed (bool, optional): Use convolutional feature masking
            reward_pred (bool, optional): Use reward prediction as auxiliary objective
        """
        super().__init__(input_shape)
        num_views = input_shape[0]
        in_chans = input_shape[1]
        img_size = input_shape[2:4]

        # --- Normalization Override ---
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # --- Encoder specifics ---
        embed_cls = MultiViewConvEmbed if conv_embed else MultiViewPatchEmbed
        self.patch_embed = embed_cls(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Output shape
        self._output_shape = (num_views, num_patches * embed_dim)
        self._num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.view_embed = nn.Parameter(torch.zeros(1, num_views, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self._norm = norm_layer(embed_dim)

        # --- Decoder specifics ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.decoder_view_embed = nn.Parameter(
            torch.zeros(1, num_views, 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        # --- Reward specifics ---
        self.reward_pred = False
        if reward_pred:
            self.reward_pred = True
            self.reward_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_reward_pred = nn.Linear(decoder_embed_dim, 1, bias=True)

        # Initialize all the weights
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        # as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.view_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_view_embed, std=0.02)
        if self.reward_pred:
            torch.nn.init.normal_(self.reward_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, V = x.shape[:2]
        x = self._color_norm(
            x / 255.0,
        )
        x, _, _ = self.forward_encoder(x, mask_ratio=0.0, num_mask_views=0)
        x = x[:, 1:]  # remove <CLS> token
        x = x.reshape([B, V, -1])  # [B, V, L*D]
        return x

    def forward_encoder(self, x, mask_ratio, num_mask_views):
        # x: [B, V, C, H, W]
        x = self.patch_embed(x)  # Embed to [B, V, L, C]

        # Add pos embed w/o cls token: [1, 1, L, C]
        x = x + self.pos_embed[:, 1:, :].unsqueeze(1)

        # Add view embed: [1, V, 1, C]
        x = x + self.view_embed

        # masking: length -> length * mask_ratio
        if mask_ratio != 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, num_mask_views)
        else:
            # reshape to [B, V*L, D]
            x = x.reshape([x.shape[0], -1, x.shape[-1]])
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self._norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if ids_restore is not None:
            # Append mask tokens to sequence -> [N, V * L, D]
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
            )
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
            )  # unshuffle
        else:
            x_ = x[:, 1:]

        # [N, V*L, D] -> [N, V, L, D]
        L = self.decoder_pos_embed.shape[1] - 1
        x_ = torch.stack(torch.split(x_, L, 1), 1)

        # Add pos embed w/o cls token: [1, 1, L, C]
        x_ = x_ + self.decoder_pos_embed[:, 1:, :].unsqueeze(1)

        # Add view embed: [1, V, 1, C]
        x_ = x_ + self.decoder_view_embed

        # [N, V, L, D] -> [N, V*L, D]
        x_ = torch.reshape(x_, [x_.shape[0], -1, x_.shape[-1]])

        # Append cls token
        cls_token = x[:, :1, :] + self.decoder_pos_embed[:, :1, :]
        x = torch.cat([cls_token, x_], dim=1)

        # Append reward token, if required
        if self.reward_pred:
            reward_token = self.reward_token.repeat(x.shape[0], 1, 1)
            x = torch.cat([x, reward_token], dim=1)

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        output = {}
        # Reward pred
        if self.reward_pred:
            reward_x = x[:, -1, :]
            reward_x = self.decoder_reward_pred(reward_x)  # [N, 1]
            output["reward"] = reward_x
            x = x[:, :-1, :]

        # Predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        # [N, V*L, D] -> [N, V, L, D]
        x = torch.stack(torch.split(x, L, 1), 1)
        output["image"] = x
        return output

    def forward_loss(self, imgs, preds):
        """
        imgs: [N, V, 3, H, W]
        preds: [N, V, L, p*p*3]
        """

        V = imgs.shape[1]
        loss = 0.0
        for v in range(V):
            target = self.patchify(imgs[:, v])
            pred = preds[:, v]
            loss_v = (pred - target) ** 2
            loss = loss + loss_v.mean()
        loss /= float(V)
        return loss

    def forward_reward_loss(self, reward, reward_preds):
        """
        reward: [N, 1]
        reward_preds: [N, 1]
        """
        x = reward_preds
        loss = ((reward - x) ** 2).mean()
        return loss

    def calculate_loss(
        self, imgs, reward=None, mask_ratio=0.9, num_mask_views=1, return_mets=False
    ):
        mets = {}
        imgs = self._color_norm(
            imgs / 255.0,
        )
        latent, _, ids_restore = self.forward_encoder(imgs, mask_ratio, num_mask_views)
        output = self.forward_decoder(latent, ids_restore)
        # Standard MAE Loss
        preds = output["image"]
        loss = self.forward_loss(imgs, preds)
        mets["mae_image_loss"] = loss.item()
        # Reward loss, if required
        if self.reward_pred:
            reward_preds = output["reward"]
            reward_loss = self.forward_reward_loss(reward, reward_preds)
            mets["mae_reward_loss"] = reward_loss.item()
            loss = loss + reward_loss
        if return_mets:
            out = (loss, mets)
        else:
            out = loss
        return out

    def report(self, imgs, mask_ratio, num_mask_views, pre_shape):
        imgs = self._color_norm(
            imgs / 255.0,
        )
        latent, _, ids_restore = self.forward_encoder(imgs, mask_ratio, num_mask_views)
        preds = self.forward_decoder(latent, ids_restore)["image"]

        with torch.no_grad():
            V = imgs.shape[1]

            # Construct visualization image
            out = []
            for v in range(V):
                img = imgs[:, v]
                pred = self.unpatchify(preds[:, v])
                out.append(torch.cat([img, pred], -2))  # height-wise concat
            out = torch.cat(out, -1)  # width-wise concat for view aggregation

            # [B*T, C, H*2, W*V]
            # pre_shape: [B, T]
            out = out.reshape(pre_shape + out.shape[1:])
            out = torch.cat(
                torch.unbind(out, 0), -1
            )  # width-wise batch aggregation, #[T, C, H*2, W*V*B]
            out = self._color_denorm(out)
            out = torch.clip(out, 0.0, 1.0)
            out = (out * 255.0).byte()
            out = out.permute(0, 2, 3, 1).cpu().numpy()  # B H W C
        return {"mae_visualization": {"video": out, "fps": 4}}

    def random_masking(self, x, mask_ratio, num_mask_views):
        """
        Perform per-sample random view masking and per-sample random masking
        For view masking, we randomly sample viewpoints and masking whole viewpoints.
        Per-sample random masking is done for remaining viewpoints after view masking.
        Per-sampling shuffling is done by argsort random noise

        Inputs:
        - x: [N, V, L, D]

        Outputs:
        - [N, LEN_KEEP, D]
        """
        if num_mask_views != 0:
            x_masked, mask, ids_restore = self._random_view_masking(
                x, mask_ratio, num_mask_views
            )
        else:
            x_masked, mask, ids_restore = self._random_uniform_masking(x, mask_ratio)

        return x_masked, mask, ids_restore

    def _random_view_masking(self, x, mask_ratio, num_mask_views):
        N, V, L, D = x.shape
        assert num_mask_views >= 1 and V > 1

        # Construct noises for view masking
        mask_view_list, noises = [], []
        for v in range(V):
            # This decides whether to mask this view or not per-sample in minibatch
            view_noise = torch.rand(N, 1, device=x.device)

            if v == 0:
                mask_view = view_noise > 0.5
                no_mask_view = ~mask_view
            else:
                curr_mask_views = torch.sum(
                    torch.cat(mask_view_list, 1), 1, keepdim=True
                )

                # M = num_mask_views
                # Find samples whose M views are already masked
                done_mask_views = curr_mask_views == num_mask_views

                # Find samples whose views should be masked to meet the number M.
                # e.g., if 1 of 3 is not masked, 2 remaining ones should be masked.
                num_should_mask_views = num_mask_views - curr_mask_views
                num_remaining_views = V - v
                must_mask_views = num_should_mask_views == num_remaining_views

                # Find samples that (i) should be masked or (ii) randomly view-masked
                mask_view = must_mask_views | (~must_mask_views & (view_noise > 0.5))

                # Filter out samples that are alredy masked
                mask_view = ~done_mask_views & mask_view
                no_mask_view = ~mask_view

            # Noises that will be used for samples whose v-viewpoint is not masked
            uniform_noise = torch.rand(N, L, device=x.device)
            # Noises that will be used for samples whose v-viewpoint is masked
            # This is set as 1 because noises with high values will be masked
            view_masked_noise = torch.ones(N, L, device=x.device) + 1e-2

            # Construct per-sample noises
            dtype = uniform_noise.dtype
            noise = (
                mask_view.to(dtype) * view_masked_noise
                + no_mask_view.to(dtype) * uniform_noise
            )

            mask_view_list.append(mask_view)
            noises.append(noise)
        noise = torch.cat(noises, 1)

        # We should adjust the effective masking ratio with view-masking
        # For instance, when we use view-masking of 1 with 2 viewpoints
        # 50% will be already masked with view-masking,
        # 40% remaining masks will be sampled from remaining 1 viewpoint,
        # so 90% masking ratio becomes effectively 80% in terms of a single viewpoint
        # so we have to adjust the effective masking ratio. This ensures that
        # uniform masking with mask_ratio is applied to unmasked viewpoint
        mask_ratio = mask_ratio + (1.0 - mask_ratio) * (num_mask_views / V)
        len_keep = int(V * L * (1 - mask_ratio))

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # make input be [N, V * L, D]
        x = x.reshape([N, V * L, D])
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, V * L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # make mask be [N, V, L]
        mask = mask.reshape([N, V, L])

        return x_masked, mask, ids_restore

    def _random_uniform_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, V, L, D]
        """
        # Convert to [N, V * L, D]
        N, V, L, D = x.shape
        x = torch.reshape(x, [N, V * L, D])

        VL = V * L
        len_keep = int(VL * (1 - mask_ratio))

        noise = torch.rand(N, VL, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, VL], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.reshape([N, V, L])

        return x_masked, mask, ids_restore

    def _color_norm(self, im):
        """Performs per-channel normalization."""
        mean = EncoderMultiViewVisionTransformer._MEAN
        std = EncoderMultiViewVisionTransformer._STD
        for i in range(3):
            im[..., i, :, :] = (im[..., i, :, :] - mean[i]) / std[i]
        return im

    def _color_denorm(self, im):
        """Performs per-channel denormalization."""
        mean = EncoderMultiViewVisionTransformer._MEAN
        std = EncoderMultiViewVisionTransformer._STD
        for i in range(3):
            im[..., i, :, :] = im[..., i, :, :] * std[i] + mean[i]
        return im

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        c = imgs.shape[1]
        h, w = self.patch_embed.grid_size
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h, w = self.patch_embed.grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def num_patches(self):
        # number of patches per each view
        return self._num_patches

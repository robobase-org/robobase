from abc import ABC
from typing import Tuple, Optional

import numpy as np
import torch
from timm.models.vision_transformer import Block
from torch import nn as nn

from robobase.models.core import RoboBaseModule


class FusionModule(RoboBaseModule, ABC):
    def __init__(
        self,
        input_shape: Tuple[int, int],
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_cameras = input_shape[0]
        self.token_size = input_shape[1]


class FusionMultiCamFeatureAttention(FusionModule):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        hidden_size: int = 128,
        depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        # V, F, v is view, F is feats (token size)
        super().__init__(input_shape)
        self._hidden_size = hidden_size
        self._views_to_keep = 1
        self._pos_embed_encode = nn.Parameter(
            torch.zeros(1, self.num_cameras, hidden_size), requires_grad=True
        )
        self._pos_embed_decode = nn.Parameter(
            torch.zeros(1, self.num_cameras, hidden_size), requires_grad=True
        )

        self._encoder_prior_embed = nn.Linear(self.token_size, hidden_size)

        self._enc_blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self._norm = norm_layer(hidden_size)

        self._decoder_blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self._mask_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size), requires_grad=True
        )
        self._decoder_norm = norm_layer(hidden_size)
        self._decoder_pred = nn.Linear(hidden_size, self.token_size, bias=True)

    def _generate_shuffle_ids_from_noise(
        self, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the shuffle indices given random noise.

        Args:
            noise: A random noise tensor.

        Returns:
            shuffle indices, restoration indices.
        """
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        return ids_shuffle, ids_restore

    def get_encoder_params(self):
        return (
            list(self._encoder_prior_embed.parameters())
            + [self._pos_embed_encode]
            + list(self._enc_blocks.parameters())
            + list(self._norm.parameters())
        )

    def forward(self, x):
        x, _, _ = self.forward_encoder(x, masking=False)
        x = x[:, 0]  # remove view axis
        assert (
            x.shape[1:] == self.output_shape
        ), f"Expected output {self.output_shape}, but got {x.shape[1:]}"
        return x

    def forward_encoder(self, x, masking=False):
        bs = x.shape[0]
        x = self._encoder_prior_embed(x)
        x = x + self._pos_embed_encode
        if masking:
            noise = torch.rand(bs, self.num_cameras, 1, device=x.device)
            mask_view_id = torch.randint(0, self.num_cameras, size=())
            noise[:, mask_view_id] = noise[:, mask_view_id] * 0.0
            ids_shuffle, ids_restore = self._generate_shuffle_ids_from_noise(noise)

            # keep the first subset
            ids_keep = ids_shuffle[:, : self._views_to_keep]
            x_masked = torch.gather(
                x, dim=1, index=ids_keep.repeat(1, 1, self._hidden_size)
            )
            x = x_masked

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([bs, self.num_cameras, 1], device=x.device)
            mask[:, : self._views_to_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            mask, ids_restore = None, None

        # apply Transformer blocks
        for blk in self._enc_blocks:
            x = blk(x)
        x = self._norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        mask_tokens = self._mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(
            x, dim=1, index=ids_restore.repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = x + self._pos_embed_decode
        # apply Transformer blocks
        for blk in self._decoder_blocks:
            x = blk(x)
        x = self._decoder_norm(x)
        x = self._decoder_pred(x)
        return x

    def forward_loss(self, feats, preds, mask) -> Optional[Tuple[torch.Tensor, dict]]:
        # Target features should be detached as a label for masked prediction.
        feats_detach = feats.detach()
        loss = (preds - feats_detach) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask[..., 0]).sum() / mask.sum()
        return loss

    def calculate_loss(self, rgb_feats):
        x, mask, ids_restore = self.forward_encoder(rgb_feats, masking=True)
        preds = self.forward_decoder(x, ids_restore)
        loss = self.forward_loss(rgb_feats, preds, mask)
        return loss

    @property
    def output_shape(self):
        return (self._hidden_size,)


class FusionMultiCamFeature(FusionModule):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        mode: str = "flatten",
    ):
        """
        Class that aggregates multi-camera features without parameters.

        Args:
            input_shape (Tuple[int, int]): Shape of the inputs: num_views, feature_dim.
            mode (str, optional): Specifies how to aggregate multi-cam features.
              available options: [flatten, average, sum]
              - flatten: flattens multi-view features (default)
              - average: averages multi-view features
              - sum: sums multi-view features
        """
        super().__init__(input_shape)
        if mode == "flatten":
            self._output_shape = (np.prod(input_shape),)
        elif mode in ["average", "sum"]:
            self._output_shape = (input_shape[1],)
        else:
            raise ValueError("Mode {mode} is not supported.")
        self._mode = mode

    def forward(self, x):
        if self._mode == "flatten":
            x = x.flatten(-2)
        elif self._mode == "average":
            x = x.mean(-2)
        elif self._mode == "sum":
            x = x.sum(-2)
        return x

    @property
    def output_shape(self):
        return self._output_shape

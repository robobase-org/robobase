from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import PatchEmbed
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed


class MultiViewPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ):
        """
        This is a class for PatchEmbed for multi-view inputs.
        Attributes are set to be compatible with PatchEmbed from timm.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.img_size = self.patch_embed.img_size
        self.patch_size = self.patch_embed.patch_size
        self.grid_size = self.patch_embed.grid_size

    def forward(self, x):
        # x: [B, V, C, H, W], where V is the number of viewpoints
        # Make sure that x has the above shape
        assert len(x.shape) == 5
        num_views = x.shape[1]

        # Convert to [B * V, C, H, W]
        x = torch.cat(torch.split(x, num_views, dim=1), dim=0)

        # Embed patches to [B * V, L, C]
        x = self.patch_embed(x)

        # Convert to [B, V, L, C]
        x = torch.stack(torch.split(x, num_views, dim=0), dim=1)
        return x


class MultiViewConvEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ):
        """
        This is a class for ConvEmbed for multi-view inputs.
        Attributes are set to be compatible with PatchEmbed from timm.
        """
        super().__init__()
        # Check whether patch size is the power of 2
        assert (patch_size & (patch_size - 1)) == 0
        num_layers = int(np.log2(patch_size))

        if not isinstance(img_size, tuple):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"img size {img_size} should be divided by patch size {patch_size}"
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])

        # Construct early convolution layers
        layers = []
        for i in range(num_layers):
            out_chans = embed_dim // (2 ** (num_layers - i))
            layers += [
                nn.Conv2d(in_chans, out_chans, 4, 2, 1),
                nn.ReLU(),
            ]
            in_chans = out_chans
        self.layers = nn.Sequential(*layers)

        # This is for compatibility with the PatchEmbed
        self.proj = nn.Conv2d(out_chans, embed_dim, 1, 1)

    def forward(self, x):
        # x: [B, V, C, H, W], where V is the number of viewpoints
        # Make sure that x has the above shape
        assert len(x.shape) == 5
        B, V = x.shape[:2]

        # Convert to [B * V, C, H, W]
        x = torch.reshape(x, ([B * V, *x.shape[2:]]))

        # Process through early convolution layers
        x = self.layers(x)
        x = self.proj(x)

        # Flatten to [B * V, C, L]
        x = x.flatten(2)

        # [B * V, C, L] -> [B * V, L, C]
        x = x.transpose(1, 2)

        # Convert to [B, V, L, C]
        x = torch.reshape(x, ([B, V, *x.shape[1:]]))
        return x


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], cls_token: bool = False
):
    """Generate 2d SinCos positional embedding

    Args:
        embed_dim: int of the embedding dimension
        grid_size: int or tuple of integers for the grid height and width
        cls_token: bool whether to include cls_token dimension or not

    Returns:
        pos_embed: [g*g, embed_dim] or[1+g*g, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    pos_embed = build_sincos2d_pos_embed(grid_size, embed_dim, interleave_sin_cos=True)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


"""
For distributional critic: https://arxiv.org/pdf/1707.06887.pdf
"""


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


def from_categorical(
    distribution, limit=20, offset=0.0, logits=True, transformation=True
):
    distribution = distribution.float().squeeze(-1)  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    shift = limit * 2 / (num_atoms - 1)
    weights = (
        torch.linspace(
            -(num_atoms // 2), num_atoms // 2, num_atoms, device=distribution.device
        )
        .float()
        .unsqueeze(-1)
    )
    if transformation:
        out = signed_parabolic((distribution @ weights) * shift) - offset
    else:
        out = (distribution @ weights) * shift - offset
    return out


def to_categorical(value, limit=20, offset=0.0, num_atoms=251, transformation=True):
    value = value.float() + offset  # Avoid any fp16 shenanigans
    shift = limit * 2 / (num_atoms - 1)
    if transformation:
        value = signed_hyperbolic(value) / shift
    else:
        value = value / shift
    value = value.clamp(-(num_atoms // 2), num_atoms // 2)
    distribution = torch.zeros(value.shape[0], num_atoms, 1, device=value.device)
    lower = value.floor().long() + num_atoms // 2
    upper = value.ceil().long() + num_atoms // 2
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-2, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-2, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


class ImgChLayerNorm(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def layernorm_for_cnn(num_channels):
    return nn.GroupNorm(1, num_channels)


def identity_cls(num_channels):
    return nn.Identity()

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from robobase.models.utils import ImgChLayerNorm, layernorm_for_cnn, identity_cls


class RoboBaseModule(nn.Module, ABC):
    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()

    def calculate_loss(self, *args, **kwargs) -> Optional[torch.Tensor]:
        return None


def get_activation_fn_from_str(act: str) -> type[nn.Module]:
    if act == "relu":
        return nn.ReLU
    elif act == "lrelu":
        return nn.LeakyReLU
    elif act == "elu":
        return nn.ELU
    elif act == "tanh":
        return nn.Tanh
    elif act == "prelu":
        return nn.PReLU
    elif act == "silu":
        return nn.SiLU
    elif act == "gelu":
        return nn.GELU
    elif act == "glu":
        return nn.GLU
    else:
        raise ValueError("%s not recognized." % act)


def get_normalization_fn_from_str(norm: str) -> type[nn.Module]:
    if norm == "layer":
        return nn.LayerNorm
    elif norm == "layer_for_cnn":
        return layernorm_for_cnn
    elif norm == "img_ch_layer":
        return ImgChLayerNorm
    elif norm == "group":
        return nn.GroupNorm
    elif norm == "batch1d":
        return nn.BatchNorm1d
    elif norm == "batch2d":
        return nn.BatchNorm2d
    elif norm == "identity":
        return identity_cls
    else:
        raise ValueError("%s not recognized." % norm)

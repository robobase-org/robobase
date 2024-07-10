import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from robobase.models.lix_utils.analysis_custom_autograd_functions import LearnS


class LocalSignalMixing(nn.Module):
    def __init__(
        self,
        pad,
        fixed_batch=False,
    ):
        """LIX regularization layer

        pad : float
            maximum regularization shift (maximum S)
        fixed batch : bool
            compute independent regularization for each sample (slower)
        """
        super().__init__()
        # +1 to avoid that the sampled values at the borders get smoothed with 0
        self.pad = int(math.ceil(pad)) + 1
        self.base_normalization_ratio = (2 * pad + 1) / (2 * self.pad + 1)
        self.fixed_batch = fixed_batch

    def get_random_shift(self, n, c, h, w, x):
        if self.fixed_batch:
            return torch.rand(size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        else:
            return torch.rand(size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

    def forward(self, x, max_normalized_shift=1.0):
        """
        x : Tensor
            input features
        max_normalized_shift : float
            current regularization shift in relative terms (current S)
        """
        if self.training:
            max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, "replicate")
            arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
            shift = self.get_random_shift(n, c, h, w, x)
            shift_offset = (1 - max_normalized_shift) / 2
            shift = (shift * max_normalized_shift) + shift_offset
            shift *= (
                2 * self.pad + 1
            )  # can start up to idx 2*pad + 1 - ignoring the left pad
            grid = base_grid + shift
            # normalize in [-1, 1]
            grid = grid * 2.0 / (h + 2 * self.pad) - 1
            return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        else:
            return x

    def get_grid(self, x, max_normalized_shift=1.0):
        max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
        shift = self.get_random_shift(n, c, h, w, x)
        shift_offset = (1 - max_normalized_shift) / 2
        shift = (shift * max_normalized_shift) + shift_offset
        shift *= 2 * self.pad + 1
        grid = base_grid + shift
        # normalize in [-1, 1]
        grid = grid * 2.0 / (h + 2 * self.pad) - 1
        return grid


class ParameterizedReg(nn.Module):
    """Augmentation/Regularization wrapper where the strength parameterized
    and is tuned with a custom autograd function

     aug : nn.Module
         augmentation/Regularization layer
     parameter_init : float
         initial strength value
     param_grad_fn : str
         custom autograd function to tune the parameter
     param_grad_fn_args : list
         arguments for the custom autograd function
    """

    def __init__(self, aug, parameter_init, param_grad_fn, param_grad_fn_args):
        super().__init__()
        self.aug = aug
        self.P = nn.Parameter(data=torch.tensor(parameter_init))
        self.param_grad_fn_name = param_grad_fn
        if param_grad_fn == "alix_param_grad":
            self.param_grad_fn = LearnS.apply
        else:
            raise NotImplementedError
        self.param_grad_fn_args = param_grad_fn_args

    def forward(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out

    def forward_no_learn(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        return out

    def forward_no_aug(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = x
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out


class NonLearnableParameterizedRegWrapper(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.aug = aug
        assert isinstance(aug, ParameterizedReg)

    def forward(self, x):
        return self.aug.forward_no_learn(x)

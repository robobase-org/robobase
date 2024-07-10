import torch
import torch as th

import torch.nn.functional as F

from robobase.models.lix_utils.analysis_utils import (
    get_local_patches_kernel,
    extract_local_patches,
)


class LearnS(torch.autograd.Function):
    """Uses neighborhood around each feature gradient position to calculate the
    spatial divergence of the gradients, and uses it to update the param S,"""

    @staticmethod
    def forward(ctx, input, param, N, target_capped_ratio, eps):
        """
        input : Tensor
            representation to be processed (used for the gradient analysis).
        param : Tensor
            ALIX parameter S to be optimized.
        N : int
            filter size used to approximate the spatial divergence as a
            convolution (to calculate the ND scores), should be odd, >= 3
        target_capped_ratio : float
            target ND scores used to adaptively tune S
        eps : float
            small stabilization constant for the ND scores
        """
        ctx.save_for_backward(param)
        ctx.N = N
        ctx.target_capped_ratio = target_capped_ratio
        ctx.eps = eps
        return input

    @staticmethod
    def backward(ctx, dy):
        N = ctx.N
        target_capped_ratio = ctx.target_capped_ratio
        eps = ctx.eps
        dy_mean_B = dy.mean(0, keepdim=True)
        ave_dy_abs = th.abs(dy_mean_B)
        pad_Hl = (N - 1) // 2
        pad_Hr = (N - 1) - pad_Hl
        pad_Wl = (N - 1) // 2
        pad_Wr = (N - 1) - pad_Wl
        pad = (pad_Wl, pad_Wr, pad_Hl, pad_Hr)
        padded_ave_dy = F.pad(dy_mean_B, pad, mode="replicate")
        loc_patches_k = get_local_patches_kernel(kernel_size=N, device=dy.device)

        local_patches_dy = extract_local_patches(
            input=padded_ave_dy, kernel=loc_patches_k, stride=1, padding=0
        )
        ave_dy_sq = ave_dy_abs.pow(2)
        patch_normalizer = (N * N) - 1

        unbiased_sq_signal = (
            local_patches_dy.pow(2).sum(dim=2) - ave_dy_sq
        ) / patch_normalizer  # expected squared signal
        unbiased_sq_noise_signal = (local_patches_dy - dy_mean_B.unsqueeze(2)).pow(
            2
        ).sum(
            2
        ) / patch_normalizer  # 1 x C x x H x W expected squared noise

        unbiased_sqn2sig = (unbiased_sq_noise_signal) / (unbiased_sq_signal + eps)

        unbiased_sqn2sig_lp1 = th.log(1 + unbiased_sqn2sig).mean()
        param_grad = target_capped_ratio - unbiased_sqn2sig_lp1

        return dy, param_grad, None, None, None

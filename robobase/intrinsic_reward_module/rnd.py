import numpy as np
import torch
from torch.nn import functional as F

from robobase.intrinsic_reward_module.core import IntrinsicRewardModule
from robobase.intrinsic_reward_module.utils import Encoder


class RND(IntrinsicRewardModule):
    """Exploration by Random Network Distillation (RND).

    Burda, Yuri, et al. "Exploration by random network distillation."
    https://arxiv.org/pdf/1810.12894.pdf

    If pixels are used, then any low-dimensional input will not be passed into RND.
    All pixel observations are concatenated on channel axis, and assumed same shape.
    """

    def __init__(
        self, latent_dim: int = 128, lr: float = 0.001, *args, **kwargs
    ) -> None:
        """Init.

        Args:
            latent_dim: The dimension of encoding vectors.
            lr: The learning rate.
        """
        super().__init__(*args, **kwargs)
        if self.use_pixels:
            obs_shapes = [v.shape for v in self.rgb_spaces.values()]
            # Fuse num views and time into channel axis
            obs_shape = (len(obs_shapes) * np.prod(obs_shapes[0][:2]),) + obs_shapes[0][
                2:
            ]
        else:
            obs_shape = self.low_dim_space.shape[-1:]
        self.predictor = Encoder(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
        ).to(self.device)
        self.target = Encoder(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def compute_irs(
        self, batch: dict[str, torch.Tensor], step: int = 0
    ) -> torch.Tensor:
        """See Base."""
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1.0 - self.kappa, step)
        next_obs = self._extract_obs(
            batch, r"rgb.*tp1" if self.use_pixels else "low_dim_state_tp1"
        )

        with torch.no_grad():
            src_feats = self.predictor(next_obs)
            tgt_feats = self.target(next_obs)
            dist = F.mse_loss(src_feats, tgt_feats, reduction="none").mean(
                dim=1, keepdim=True
            )
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
            intrinsic_rewards = dist

        return intrinsic_rewards * beta_t

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        """See Base."""
        obs = self._extract_obs(
            batch, r"rgb(?!.*?tp1)" if self.use_pixels else "low_dim_state"
        )
        src_feats = self.predictor(obs)
        with torch.no_grad():
            tgt_feats = self.target(obs)
        self.opt.zero_grad()
        loss = F.mse_loss(src_feats, tgt_feats)
        loss.backward()
        self.opt.step()

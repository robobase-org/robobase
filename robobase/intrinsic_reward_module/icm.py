import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from robobase.intrinsic_reward_module.core import IntrinsicRewardModule
from robobase.intrinsic_reward_module.utils import Encoder


class InverseDynamicsModel(nn.Module):
    """Inverse model for reconstructing transition process."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        """Init.

        Args:
            latent_dim: The dimension of encoding vectors of the observations.
            action_dim: The dimension of predicted actions.

        Returns:
            Model instance.
        """
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2 * latent_dim, 256), nn.ReLU(), nn.Linear(256, action_dim)
        )

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs: Current observations.
            next_obs: Next observations.

        Returns:
            Predicted actions.
        """
        return self.trunk(torch.cat([obs, next_obs], dim=1))


class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transition process."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        """Init.

        Args:
            latent_dim: The dimension of encoding vectors of the observations.
            action_dim: The dimension of predicted actions.

        Returns:
            Model instance.
        """
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: torch.Tensor, pred_actions: torch.Tensor) -> torch.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs: Current observations.
            pred_actions: Predicted observations.

        Returns:
            Predicted next-obs.
        """
        return self.trunk(torch.cat([obs, pred_actions], dim=1))


class ICM(IntrinsicRewardModule):
    """Curiosity-Driven Exploration by Self-Supervised Prediction.

    Pathak, Deepak, et al. "Curiosity-driven exploration by self-supervised prediction."
    International conference on machine learning. PMLR, 2017.
    https://arxiv.org/abs/1705.05363

    If pixels are used, then any low-dimensional input will not be passed into ICM.
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
        self.encoder = Encoder(
            obs_shape=obs_shape,
            latent_dim=latent_dim,
        ).to(self.device)

        action_dim = np.prod(self.action_space.shape)
        self.im = InverseDynamicsModel(latent_dim=latent_dim, action_dim=action_dim).to(
            self.device
        )
        self.fm = ForwardDynamicsModel(latent_dim=latent_dim, action_dim=action_dim).to(
            self.device
        )
        self.dynamics_opt = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.im.parameters())
            + list(self.fm.parameters()),
            lr=lr,
        )

    def compute_irs(
        self, batch: dict[str, torch.Tensor], step: int = 0
    ) -> torch.Tensor:
        """See Base."""
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1.0 - self.kappa, step)
        obs = self._extract_obs(
            batch, r"rgb(?!.*?tp1)" if self.use_pixels else "low_dim_state"
        )
        next_obs = self._extract_obs(
            batch, r"rgb.*tp1" if self.use_pixels else "low_dim_state_tp1"
        )
        action = batch["action"].view(obs.shape[0], -1)
        encoded_obs = self.encoder(obs)
        encoded_next_obs = self.encoder(next_obs)
        pred_next_obs = self.fm(encoded_obs, action)
        intrinsic_rewards = torch.linalg.vector_norm(
            encoded_next_obs - pred_next_obs, ord=2, dim=1, keepdim=True
        )
        return intrinsic_rewards * beta_t

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        """See Base."""
        obs = self._extract_obs(
            batch, r"rgb(?!.*?tp1)" if self.use_pixels else "low_dim_state"
        )
        next_obs = self._extract_obs(
            batch, r"rgb.*tp1" if self.use_pixels else "low_dim_state_tp1"
        )
        action = batch["action"].view(obs.shape[0], -1)
        self.dynamics_opt.zero_grad()
        encoded_obs = self.encoder(obs)
        encoded_next_obs = self.encoder(next_obs)
        pred_actions = self.im(encoded_obs, encoded_next_obs)
        im_loss = F.mse_loss(pred_actions, action)
        pred_next_obs = self.fm(encoded_obs, action)
        fm_loss = F.mse_loss(pred_next_obs, encoded_next_obs)
        (im_loss + fm_loss).backward()
        self.dynamics_opt.step()

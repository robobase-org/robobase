import torch
from torch import nn


class Encoder(nn.Module):
    """Encoder for encoding observations."""

    def __init__(self, obs_shape: tuple, latent_dim: int) -> None:
        """Init.

        Args:
            obs_shape: The data shape of observations.
            latent_dim: The dimension of encoding vectors.

        Returns:
            Encoder instance.
        """
        super().__init__()
        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.ones(size=tuple(obs_shape))
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.linear = nn.Linear(n_flatten, latent_dim)
        else:
            self.trunk = nn.Sequential(nn.Linear(obs_shape[0], 256), nn.ReLU())
            self.linear = nn.Linear(256, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the input tensors.

        Args:
            obs: Observations.

        Returns:
            Encoding tensors.
        """
        if len(obs.shape) == 4:
            # RGB image
            obs = obs.float() / 255.0 - 0.5
        return self.linear(self.trunk(obs))

from abc import ABC, abstractmethod
import torch
from gymnasium import spaces

from robobase.method.utils import (
    extract_many_from_spec,
    extract_from_spec,
    extract_many_from_batch,
)


class IntrinsicRewardModule(ABC):
    """Base class of intrinsic reward module."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: torch.device,
        beta: float = 0.05,
        kappa: float = 0.000025,
    ) -> None:
        """Init.

        Args:
            observation_space: The observation space of environment.
            action_space: The action space of environment.
            device: Device to run the model.
            beta: The initial weighting coefficient of the intrinsic rewards.
            kappa: The decay rate.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.beta = beta
        self.kappa = kappa

        self.rgb_spaces = extract_many_from_spec(
            observation_space, r"rgb.*", missing_ok=True
        )
        self.low_dim_space = extract_from_spec(
            observation_space, "low_dim_state", missing_ok=True
        )
        self.use_pixels = len(self.rgb_spaces) > 0

    @abstractmethod
    def compute_irs(
        self, batch: dict[str, torch.Tensor], step: int = 0
    ) -> torch.Tensor:
        """Compute the intrinsic rewards for current samples.

        Args:
            batch: Batch of data.
            step: The global training step.

        Returns:
            The intrinsic rewards.
        """

    @abstractmethod
    def update(
        self,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            batch: Batch of data.
        """

    def _extract_obs(self, batch: dict[str, torch.Tensor], name_or_regex: str):
        if self.use_pixels:
            # dict of {"cam_name": (B, T, 3, H, W)}
            obs = extract_many_from_batch(batch, name_or_regex)
            # Fold views into time axis
            obs = torch.cat(list(obs.values()), 1)
            # Fold time into channel axis
            obs = obs.view(obs.shape[0], -1, *obs.shape[3:])
        else:
            # Get last timestep
            obs = batch[name_or_regex][:, -1]
        return obs

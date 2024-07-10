from functools import partial
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer

from robobase.models.fusion import FusionModule
from robobase.models.encoder import EncoderModule
from robobase.models.fully_connected import FullyConnectedModule

from robobase.method.core import Method
from diffusers.optimization import get_scheduler

from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    extract_from_batch,
    extract_many_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
    loss_weights,
)

from typing import Optional, Iterator


class Actor(nn.Module):
    def __init__(
        self,
        action_space: spaces.Box,
        actor_model: FullyConnectedModule,
    ):
        super().__init__()
        assert len(action_space.shape) == 2
        self.action_space = action_space
        self.actor = actor_model
        self.sequence_length = action_space.shape[0]
        self.action_dim = action_space.shape[1]

    @property
    def preferred_optimiser(self) -> callable:
        return getattr(
            self.actor,
            "preferred_optimiser",
            partial(torch.optim.Adam, self.parameters()),
        )

    def _combine(self, low_dim_obs, fused_view_feats):
        flat_feats = []
        if low_dim_obs is not None:
            flat_feats.append(low_dim_obs)
        if fused_view_feats is not None:
            flat_feats.append(fused_view_feats)
        obs_features = torch.cat(flat_feats, dim=-1)
        return obs_features

    def forward(self, low_dim_obs, fused_view_feats) -> torch.Tensor:
        obs_features = self._combine(low_dim_obs, fused_view_feats)
        net_ins = {
            "features": obs_features,
        }
        return self.actor(net_ins)


class BC(Method):
    """A very simple BC method that can be used as a base for others IL algorithms."""

    def __init__(
        self,
        lr: float,
        adaptive_lr: bool,
        num_train_steps: int,
        actor_grad_clip: Optional[float] = None,
        actor_model: Optional[FullyConnectedModule] = None,
        encoder_model: Optional[EncoderModule] = None,
        view_fusion_model: Optional[FusionModule] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.adaptive_lr = adaptive_lr
        self.num_train_steps = num_train_steps
        self.actor_grad_clip = actor_grad_clip
        self.actor_model = actor_model
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        # T should be same across all obs
        self.time_dim = list(self.observation_space.values())[0].shape[0]
        self.use_pixels = len(self.rgb_spaces) > 0
        self.use_multicam_fusion = len(self.rgb_spaces) > 1
        self.actor = self.encoder = self.view_fusion = None
        self.build_encoder()
        self.build_view_fusion()
        self.build_actor()

        self.lr_scheduler = None
        if self.adaptive_lr:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.actor_opt,
                num_warmup_steps=100,
                num_training_steps=num_train_steps,
            )

    @property
    def time_obs_size(self) -> int:
        time_obs_spec = extract_from_spec(
            self.observation_space, "time", missing_ok=True
        )
        time_obs_size = 0
        if time_obs_spec is not None:
            time_obs_size = time_obs_spec.shape[1]
        return time_obs_size

    @property
    def low_dim_size(self) -> int:
        low_dim_state_spec = extract_from_spec(
            self.observation_space, "low_dim_state", missing_ok=True
        )
        low_dim_in_size = 0
        if low_dim_state_spec is not None:
            low_dim_in_size = low_dim_state_spec.shape[1] * low_dim_state_spec.shape[0]
        return low_dim_in_size

    def build_actor(self):
        self.actor_model = self.actor_model(
            input_shapes=self.get_fully_connected_inputs(),
            output_shape=self.action_space.shape[-1],
            num_envs=self.num_train_envs + 1,  # +1 for eval
        )
        self.actor = Actor(self.action_space, self.actor_model).to(self.device)
        self.actor_opt = (self.actor.preferred_optimiser)(lr=self.lr)

    def build_encoder(self):
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            rgb_shapes = [s.shape for s in rgb_spaces.values()]
            assert np.all(
                [sh == rgb_shapes[0] for sh in rgb_shapes]
            ), "Expected all RGB obs to be same shape."

            num_views = len(rgb_shapes)
            # Multiply first two dimensions to consider frame stacking
            obs_shape = (np.prod(rgb_shapes[0][:2]), *rgb_shapes[0][2:])
            self.encoder = self.encoder_model(input_shape=(num_views, *obs_shape))
            self.encoder.to(self.device)
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

    def build_view_fusion(self):
        self.rgb_latent_size = 0
        if not self.use_pixels:
            return
        if self.use_multicam_fusion:
            if self.view_fusion_model is None:
                logging.warn(
                    "Multicam fusion is enabled but view_fusion_model is not set!"
                )
                self.view_fusion_opt = None
                return

            self.view_fusion = self.view_fusion_model(
                input_shape=self.encoder.output_shape
            )
            self.view_fusion.to(self.device)
            self.view_fusion_opt = None
            if len([_ for _ in self.view_fusion.parameters()]) != 0:
                # Introduce optimizer when view_fusion_model is parametrized
                self.view_fusion_opt = torch.optim.Adam(
                    self.view_fusion.parameters(), lr=self.lr
                )
            self.rgb_latent_size = self.view_fusion.output_shape[-1]
        else:
            self.view_fusion = lambda x: x[:, 0]
            self.rgb_latent_size = self.encoder.output_shape[-1]

    def get_fully_connected_inputs(self) -> dict[str, tuple]:
        input_sizes = {}
        obs_features_size = 0
        if self.rgb_latent_size > 0:
            obs_features_size += self.rgb_latent_size
        if self.low_dim_size > 0:
            obs_features_size += self.low_dim_size
        input_sizes["features"] = (obs_features_size,)
        if not self.frame_stack_on_channel and self.time_dim > 0:
            for k, v in input_sizes.items():
                input_sizes[k] = (self.time_dim,) + v
        return input_sizes

    def extract_low_dim_state(self, batch: dict[str, torch.Tensor]):
        low_dim_obs = flatten_time_dim_into_channel_dim(
            extract_from_batch(batch, "low_dim_state")
        )
        next_low_dim_obs = flatten_time_dim_into_channel_dim(
            extract_from_batch(batch, "low_dim_state_tp1")
        )
        return low_dim_obs, next_low_dim_obs

    def extract_pixels(self, batch: dict[str, torch.Tensor]):
        # dict of {"cam_name": (B, V, T, 3, H, W)}
        rgb_obs_dict = extract_many_from_batch(batch, r"rgb(?!.*?tp1)")
        # Convert to tensor of (B, VIEWS, ENC_DIM)
        rgb_obs = flatten_time_dim_into_channel_dim(
            stack_tensor_dictionary(rgb_obs_dict, 1),
            has_view_axis=True,
        )
        metrics = {}
        if self.logging:
            # Get first batch item and last timestep
            for k, v in rgb_obs_dict.items():
                metrics[k] = v[0, -1]
        next_rgb_obs = flatten_time_dim_into_channel_dim(
            stack_tensor_dictionary(extract_many_from_batch(batch, r"rgb.*tp1"), 1),
            has_view_axis=True,
        )
        return rgb_obs, next_rgb_obs, metrics

    def update_actor(self, low_dim_obs, fused_view_feats, action, loss_coeff):
        metrics = dict()
        action_pred = self.actor(low_dim_obs, fused_view_feats)
        # TOOD check trajectory horrizon
        mse_loss = (
            F.mse_loss(action_pred, action, reduction="none")
            .mean(-1)
            .mean(-1, keepdims=True)
        )
        actor_loss = (mse_loss * loss_coeff.unsqueeze(1)).mean()

        new_pri = torch.sqrt(mse_loss + 1e-10)
        self._new_priority = (new_pri / torch.max(new_pri)).cpu().detach().numpy()

        # calculate gradient
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)

        # step optimizer
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()
        if self.use_pixels and self.encoder is not None:
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()

        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.logging:
            metrics["actor_loss"] = actor_loss.item()
        return metrics

    def encode(self, rgb_obs):
        metrics = dict()

        # Update the encoder with auxiliary objective if we need to.
        metrics.update(self.update_encoder_rep(rgb_obs.float()))

        # Extract the features from the encoder
        multi_view_rgb_feats = self.encoder(rgb_obs.float())

        return metrics, multi_view_rgb_feats

    def multi_view_fusion(self, rgb_obs, multi_view_feats):
        metrics = dict()

        if self.use_multicam_fusion:
            # Update the view fusion with auxiliary objective if we need to.
            view_fusion_metrics = self.update_view_fusion_rep(multi_view_feats)
            metrics.update(view_fusion_metrics)
            # Extract features from the newly updated encoder
            if len(view_fusion_metrics) != 0:
                # TODO: Find a better way to check if weights updated
                multi_view_feats = self.encoder(rgb_obs.float())

        # Fuse the multi view features (e.g., AvgPool, MLP, Identity, ..)
        fused_view_feats = self.view_fusion(multi_view_feats)

        return metrics, fused_view_feats

    def update_encoder_rep(self, rgb_obs):
        loss = self.encoder.calculate_loss(rgb_obs)
        if loss is None:
            return {}
        self.encoder.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_opt.step()
        return {"encoder_rep_loss": loss.item()}

    def update_view_fusion_rep(self, rgb_feats):
        # NOTE: This method will always try to update the encoder
        # Whether to update the encoder is the responsibility of view_fusion_model.
        # It should detach rgb_feats if it does not want to update the encoder.
        loss = self.view_fusion.calculate_loss(rgb_feats)
        if loss is None:
            return {}
        assert (
            self.view_fusion_opt is not None
        ), "Use `update_encoder_rep` to only update the encoder parameters."
        self.encoder.zero_grad(set_to_none=True)
        self.view_fusion_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_opt.step()
        self.view_fusion_opt.step()
        return {"view_fusion_rep_loss": loss.item()}

    def _act(self, observations: dict[str, torch.Tensor], eval_mode: bool):
        low_dim_obs = fused_rgb_feats = None
        if self.low_dim_size > 0:
            low_dim_obs = flatten_time_dim_into_channel_dim(
                extract_from_spec(observations, "low_dim_state")
            )
        if self.use_pixels:
            rgb_obs = flatten_time_dim_into_channel_dim(
                stack_tensor_dictionary(
                    extract_many_from_spec(observations, r"rgb.*"), 1
                ),
                has_view_axis=True,
            )
            with torch.no_grad():
                multi_view_rgb_feats = self.encoder(rgb_obs.float())
                fused_rgb_feats = self.view_fusion(multi_view_rgb_feats)
        actions = self.actor(low_dim_obs, fused_rgb_feats)
        return actions.detach()

    def act(self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool):
        with torch.no_grad():
            return self._act(observations, eval_mode)

    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        metrics = dict()
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        action = batch["action"]
        loss_coeff = loss_weights(batch, self.replay_beta)
        low_dim_obs = None
        fused_view_feats = None
        if self.low_dim_size > 0:
            low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)

        if self.use_pixels:
            rgb_obs, _, ext_metrics = self.extract_pixels(batch)
            metrics.update(ext_metrics)
            enc_metrics, rgb_feats = self.encode(rgb_obs)
            metrics.update(enc_metrics)
            (
                fusion_metrics,
                fused_view_feats,
            ) = self.multi_view_fusion(rgb_obs, rgb_feats)
            metrics.update(fusion_metrics)

        if fused_view_feats is not None:
            fused_view_feats = fused_view_feats.detach()
        metrics.update(
            self.update_actor(low_dim_obs, fused_view_feats, action, loss_coeff)
        )

        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            replay_buffer.set_priority(
                indices=batch["indices"].cpu().detach().numpy(),
                priorities=self._new_priority**self.replay_alpha,
            )

        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.

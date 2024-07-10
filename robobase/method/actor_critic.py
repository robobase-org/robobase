import time
from abc import ABC
from copy import deepcopy
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Distribution

from robobase import utils
from robobase.method.core import OffPolicyMethod
from robobase.models.fusion import FusionModule
from robobase.models.encoder import EncoderModule
from robobase.models.fully_connected import FullyConnectedModule
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    extract_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
    extract_many_from_batch,
    loss_weights,
)
from robobase.models.utils import from_categorical, to_categorical


class Critic(nn.Module):
    def __init__(self, critic_model: FullyConnectedModule, num_critics: int):
        super().__init__()
        self.qs = nn.ModuleList([deepcopy(critic_model) for _ in range(num_critics)])
        self.apply(utils.weight_init)
        self.distributional = False

    def forward(self, low_dim_obs, fused_view_feats, action, time_obs):
        net_ins = {"action": action.view(action.shape[0], -1)}
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        if time_obs is not None:
            net_ins["time_obs"] = time_obs
        return torch.cat(
            [q(net_ins) for q in self.qs],
            -1,
        )

    def reset(self, env_index: int):
        [q.reset(env_index) for q in self.qs]

    def set_eval_env_running(self, value: bool):
        for q in self.qs:
            q.eval_env_running = value


class DistributionalCritic(Critic):
    def __init__(self, limit, atoms, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributional = True
        self.limit = limit
        self.atoms = atoms
        self.transform = transform

    def to_dist(self, qs):
        return torch.cat(
            [
                to_categorical(
                    qs[:, q_idx].unsqueeze(-1),
                    limit=self.limit,
                    num_atoms=self.atoms,
                    transformation=self.transform,
                )
                for q_idx in range(qs.size(-1))
            ],
            dim=-1,
        )

    def from_dist(self, qs):
        return torch.cat(
            [
                from_categorical(
                    qs[..., q_idx],
                    limit=self.limit,
                    transformation=self.transform,
                )
                for q_idx in range(qs.size(-1))
            ],
            dim=-1,
        )

    def compute_distributional_critic_loss(self, qs, target_qs):
        loss = 0.0
        for q_idx in range(qs.size(-1)):
            loss += -torch.sum(
                torch.log_softmax(qs[[..., q_idx]], -1)
                * target_qs.squeeze(-1).detach(),
                -1,
            )
        return loss.unsqueeze(-1)


class Actor(nn.Module):
    def __init__(
        self,
        actor_model: FullyConnectedModule,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.apply(utils.weight_init)

    def forward(self, low_dim_obs, fused_view_feats) -> Distribution:
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        mu = self.actor_model(net_ins)
        mu = torch.tanh(mu)
        # Make deterministic
        dist = utils.TruncatedNormal(mu, torch.zeros_like(mu))
        return dist

    def reset(self, env_index: int):
        self.actor_model.reset(env_index)

    def set_eval_env_running(self, value: bool):
        self.actor_model.eval_env_running = value


class ActorCritic(OffPolicyMethod, ABC):
    def __init__(
        self,
        num_explore_steps: int,
        actor_lr: float,
        critic_lr: float,
        view_fusion_lr: float,
        encoder_lr: float,
        weight_decay: float,
        critic_target_tau: float,
        num_critics: int,
        actor_grad_clip: Optional[float],
        critic_grad_clip: Optional[float],
        always_bootstrap: bool,
        bc_lambda: float,
        action_sequence_network_type: str,
        actor_model: Optional[FullyConnectedModule],
        critic_model: Optional[FullyConnectedModule],
        encoder_model: Optional[EncoderModule],
        view_fusion_model: Optional[FusionModule],
        critic_updates_shared_vision_encoder: bool,
        distributional_critic: bool,
        distributional_critic_limit: float,
        distributional_critic_atoms: int,
        distributional_critic_transform: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_explore_steps = num_explore_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.view_fusion_lr = view_fusion_lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.critic_target_tau = critic_target_tau
        self.num_critics = num_critics
        self.actor_grad_clip = actor_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.always_bootstrap = always_bootstrap
        self.bc_lambda = bc_lambda
        self.action_sequence_network_type = action_sequence_network_type.lower()
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model
        self.critic_updates_shared_vision_encoder = critic_updates_shared_vision_encoder
        self.distributional_critic = distributional_critic
        self.distributional_critic_limit = distributional_critic_limit
        self.distributional_critic_atoms = distributional_critic_atoms
        self.distributional_critic_transform = distributional_critic_transform
        if not self.critic_updates_shared_vision_encoder:
            raise NotImplementedError("Separate actor-critic networks not supported.")
        if self.action_sequence_network_type not in ["rnn", "mlp"]:
            raise ValueError(
                f"action_sequence_network_type: {action_sequence_network_type} "
                "not supported."
            )
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        self.use_pixels = len(self.rgb_spaces) > 0
        self.use_multicam_fusion = len(self.rgb_spaces) > 1
        # T should be same across all obs
        self.time_dim = list(self.observation_space.values())[0].shape[0]
        self.actor = self.critic = self.encoder = self.view_fusion = None
        self.build_encoder()
        self.build_view_fusion()
        self.build_actor()
        self.critic, self.critic_target, self.critic_opt = self.build_critic()
        self.intr_critic = self.intr_critic_target = self.intr_critic_opt = None
        if self.intrinsic_reward_module:
            (
                self.intr_critic,
                self.intr_critic_target,
                self.intr_critic_opt,
            ) = self.build_critic()

    def build_actor(self):
        input_shapes = self.get_fully_connected_inputs()
        if "time_obs" in input_shapes:
            # We don't use time_obs for actor
            input_shapes.pop("time_obs")
        self.actor_model = self.actor_model(
            input_shapes=input_shapes,
            output_shape=self.action_space.shape[-1],
            num_envs=self.num_train_envs + 1,  # +1 for eval
        )
        self.actor = Actor(self.actor_model).to(self.device)
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay
        )

    def build_critic(self) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
        input_shapes = self.get_fully_connected_inputs()
        input_shapes["actions"] = (np.prod(self.action_space.shape),)
        critic_model = self.critic_model(
            input_shapes=input_shapes, num_envs=self.num_train_envs + 1
        )
        if self.distributional_critic:
            critic = DistributionalCritic(
                self.distributional_critic_limit,
                self.distributional_critic_atoms,
                self.distributional_critic_transform,
                critic_model,
                self.num_critics,
            ).to(self.device)
        else:
            critic = Critic(critic_model, self.num_critics).to(self.device)
        critic_target = deepcopy(critic)
        critic_target.load_state_dict(critic.state_dict())
        critic_opt = torch.optim.AdamW(
            critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay
        )
        critic_target.eval()
        return critic, critic_target, critic_opt

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
            if self.frame_stack_on_channel:
                obs_shape = (np.prod(rgb_shapes[0][:2]), *rgb_shapes[0][2:])
            else:
                # T is folded into batch
                obs_shape = rgb_shapes[0][1:]
            self.encoder = self.encoder_model(input_shape=(num_views, *obs_shape))
            self.encoder.to(self.device)
            self.encoder_opt = torch.optim.AdamW(
                self.encoder.parameters(),
                lr=self.encoder_lr,
                weight_decay=self.weight_decay,
            )

    def build_view_fusion(self):
        self.rgb_latent_size = 0
        if not self.use_pixels:
            return
        if self.use_multicam_fusion:
            self.view_fusion = self.view_fusion_model(
                input_shape=self.encoder.output_shape
            )
            self.view_fusion.to(self.device)
            self.view_fusion_opt = None
            if len([_ for _ in self.view_fusion.parameters()]) != 0:
                # Introduce optimizer when view_fusion_model is parametrized
                self.view_fusion_opt = torch.optim.AdamW(
                    self.view_fusion.parameters(),
                    lr=self.view_fusion_lr,
                    weight_decay=self.weight_decay,
                )
            self.rgb_latent_size = self.view_fusion.output_shape[-1]
        else:
            self.view_fusion = lambda x: x[:, 0]
            self.rgb_latent_size = self.encoder.output_shape[-1]

    def get_fully_connected_inputs(self) -> dict[str, tuple]:
        input_sizes = {}
        if self.rgb_latent_size > 0:
            input_sizes["fused_view_feats"] = (self.rgb_latent_size,)
        if self.low_dim_size > 0:
            input_sizes["low_dim_obs"] = (self.low_dim_size,)
        if self.time_obs_size > 0:
            input_sizes["time_obs"] = (self.time_obs_size,)
        if not self.frame_stack_on_channel and self.time_dim > 0:
            for k, v in input_sizes.items():
                input_sizes[k] = (self.time_dim,) + v
        return input_sizes

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
            if self.frame_stack_on_channel:
                low_dim_in_size = np.prod(low_dim_state_spec.shape[-2:])
            else:
                low_dim_in_size = low_dim_state_spec.shape[-1]
        return low_dim_in_size

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

    def act(self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool):
        if step < self.num_explore_steps and not eval_mode:
            return self.random_explore_action
        else:
            with torch.no_grad():
                return self._act(observations, eval_mode)

    def _act_extract_rgb_obs(self, observations: dict[str, torch.Tensor]):
        rgb_obs = extract_many_from_spec(observations, r"rgb.*")
        rgb_obs = stack_tensor_dictionary(rgb_obs, 1)
        if self.frame_stack_on_channel:
            rgb_obs = flatten_time_dim_into_channel_dim(rgb_obs, has_view_axis=True)
        else:
            rgb_obs = rgb_obs.transpose(1, 2)
            rgb_obs = rgb_obs.view(-1, *rgb_obs.shape[2:])
        return rgb_obs

    def _act_extract_low_dim_state(self, observations: dict[str, torch.Tensor]):
        low_dim_obs = extract_from_spec(observations, "low_dim_state")
        if self.frame_stack_on_channel:
            low_dim_obs = flatten_time_dim_into_channel_dim(low_dim_obs)
        return low_dim_obs

    def _act(self, observations: dict[str, torch.Tensor], eval_mode: bool):
        low_dim_obs = fused_rgb_feats = None
        if self.low_dim_size > 0:
            low_dim_obs = self._act_extract_low_dim_state(observations)
        if self.use_pixels:
            rgb_obs = self._act_extract_rgb_obs(observations)
            with torch.no_grad():
                multi_view_rgb_feats = self.encoder(rgb_obs.float())
                fused_rgb_feats = self.view_fusion(multi_view_rgb_feats)
                if not self.frame_stack_on_channel:
                    fused_rgb_feats = fused_rgb_feats.view(
                        -1, self.time_dim, *fused_rgb_feats.shape[1:]
                    )
        dist = self.actor(low_dim_obs, fused_rgb_feats)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        return action

    def update_critic(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        reward,
        discount,
        bootstrap,
        next_low_dim_obs,
        next_fused_view_feats,
        step,
        time_obs,
        next_time_obs,
        loss_coeff,
        updating_intrinsic_critic,
    ):
        lp = "intrinsic_" if updating_intrinsic_critic else ""
        if updating_intrinsic_critic:
            critic, critic_opt = (
                self.intr_critic,
                self.intr_critic_opt,
            )
        else:
            critic, critic_opt = (
                self.critic,
                self.critic_opt,
            )

        metrics = dict()
        target_qs = self.calculate_target_q(
            step,
            next_low_dim_obs,
            next_fused_view_feats,
            next_time_obs,
            reward,
            discount,
            bootstrap,
            updating_intrinsic_critic,
        )

        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)

        if self.distributional_critic:
            q_critic_loss = critic.compute_distributional_critic_loss(qs, target_qs)
        else:
            target_qs = target_qs.repeat(1, self.num_critics)
            q_critic_loss = F.mse_loss(qs, target_qs, reduction="none").mean(
                -1, keepdim=True
            )
        critic_loss = q_critic_loss * loss_coeff.unsqueeze(1)

        # Compute priority
        new_pri = torch.sqrt(q_critic_loss + 1e-10)
        self._td_error = (new_pri / torch.max(new_pri)).cpu().detach().numpy()
        critic_loss = torch.mean(critic_loss)

        if self.logging:
            metrics[f"{lp}critic_target_q"] = target_qs.mean().item()
            for i in range(1, self.num_critics):
                metrics[f"{lp}critic_q{i+1}"] = qs[..., i].mean().item()
            metrics[f"{lp}critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.critic_grad_clip:
            nn.utils.clip_grad_norm_(critic.parameters(), self.critic_grad_clip)
        critic_opt.step()
        if self.use_pixels and self.encoder is not None:
            if self.critic_grad_clip:
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.critic_grad_clip
                )
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()
        return metrics

    def get_bc_loss(self, predicted_action, buffer_action, demos):
        metrics = dict()
        bc_loss = 0
        if demos is not None:
            # Only apply loss to demo items
            demos = demos.float().unsqueeze(1)
            bs = demos.shape[0]
            if demos.sum() > 0:
                bc_loss = (
                    F.mse_loss(
                        predicted_action.view(bs, -1),
                        buffer_action.view(bs, -1),
                        reduction="none",
                    )
                    * demos
                )
                bc_loss = self.bc_lambda * bc_loss.sum() / demos.sum()
                if self.logging:
                    metrics["actor_bc_loss"] = bc_loss.item()
            if self.logging:
                metrics["ratio_of_demos"] = demos.mean().item()
        return metrics, bc_loss

    def _compute_actor_loss(
        self, low_dim_obs, fused_view_feats, action, time_obs, loss_coeff, critic
    ):
        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
        if self.distributional_critic:
            qs = critic.from_dist(qs)
        min_q = qs.min(-1, keepdim=True)[0]
        return (-min_q * loss_coeff.unsqueeze(1)).mean()

    def update_actor(
        self, low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
    ):
        metrics = dict()

        dist = self.actor(low_dim_obs, fused_view_feats)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        base_actor_loss = self._compute_actor_loss(
            low_dim_obs, fused_view_feats, action, time_obs, loss_coeff, self.critic
        )
        intr_actor_loss = 0
        if self.intrinsic_reward_module is not None:
            intr_actor_loss = self._compute_actor_loss(
                low_dim_obs,
                fused_view_feats,
                action,
                time_obs,
                loss_coeff,
                self.intr_critic,
            )
        bc_metrics, bc_loss = self.get_bc_loss(dist.mean, act, demos)
        metrics.update(bc_metrics)
        actor_loss = base_actor_loss + intr_actor_loss + bc_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()

        if self.logging:
            metrics["mean_act"] = dist.mean.mean().item()
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def extract_batch(
        self, replay_iter: Iterator[dict[str, torch.Tensor]]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        metrics = dict()
        if self.logging:
            start_time = time.time()
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self.logging:
            metrics["buffer_sample_time"] = time.time() - start_time
        action = batch["action"]
        reward = batch["reward"].unsqueeze(1)
        discount = batch["discount"].to(reward.dtype).unsqueeze(1)
        terminal = batch["terminal"].to(reward.dtype)
        truncated = batch["truncated"].to(reward.dtype)
        # 1. If not terminal and not truncated, we bootstrap
        # 2. If not terminal and truncated, we bootstrap
        # 3. If terminal and not truncated, we don't bootstrap
        # 4. If terminal and truncated,(e.g., success in last timestep)
        #    we don't bootstrap as terminal has a priortiy over truncated
        # In summary, we do not bootstrap when terminal; otherwise we do bootstrap
        bootstrap = (1.0 - terminal).unsqueeze(1)
        if self.always_bootstrap:
            # Override bootstrap to be 1
            bootstrap = torch.ones_like(bootstrap)

        time_obs = extract_from_batch(batch, "time", missing_ok=True)
        next_time_obs = extract_from_batch(batch, "time_tp1", missing_ok=True)
        if time_obs is not None:
            time_obs = time_obs.float()[:, -1]
            next_time_obs = next_time_obs.float()[:, -1]
        demos = extract_from_batch(batch, "demo", missing_ok=True)
        loss_coeff = loss_weights(batch, self.replay_beta)
        if self.logging:
            metrics["batch_reward"] = reward.mean().item()
        return (
            metrics,
            batch,
            action,
            reward,
            discount,
            terminal,
            truncated,
            bootstrap,
            demos,
            time_obs,
            next_time_obs,
            loss_coeff,
        )

    def extract_low_dim_state(self, batch: dict[str, torch.Tensor]):
        low_dim_obs = extract_from_batch(batch, "low_dim_state")
        next_low_dim_obs = extract_from_batch(batch, "low_dim_state_tp1")
        if self.frame_stack_on_channel:
            low_dim_obs = flatten_time_dim_into_channel_dim(low_dim_obs)
            next_low_dim_obs = flatten_time_dim_into_channel_dim(next_low_dim_obs)
        return low_dim_obs, next_low_dim_obs

    def extract_pixels(self, batch: dict[str, torch.Tensor]):
        # dict of {"cam_name": (B, T, 3, H, W)}
        rgb_obs = extract_many_from_batch(batch, r"rgb(?!.*?tp1)")
        next_rgb_obs = extract_many_from_batch(batch, r"rgb.*tp1")
        metrics = {}
        if self.logging:
            # Get first batch item and last timestep
            for k, v in rgb_obs.items():
                metrics[k] = v[0, -1]
            for k, v in next_rgb_obs.items():
                metrics[k] = v[0, -1]

        # -> (B, V, T, 3. H, W)
        rgb_obs = stack_tensor_dictionary(rgb_obs, 1)
        next_rgb_obs = stack_tensor_dictionary(next_rgb_obs, 1)
        if self.frame_stack_on_channel:
            # -> (B, V, T * 3, H, W)
            rgb_obs = flatten_time_dim_into_channel_dim(rgb_obs, has_view_axis=True)
            next_rgb_obs = flatten_time_dim_into_channel_dim(
                next_rgb_obs, has_view_axis=True
            )
        else:
            # Fold into batch. Will be unfolded later.
            # -> (B * T, V, 3. H, W)
            rgb_obs = rgb_obs.transpose(1, 2)
            rgb_obs = rgb_obs.view(-1, *rgb_obs.shape[2:])
            next_rgb_obs = next_rgb_obs.transpose(1, 2)
            next_rgb_obs = next_rgb_obs.view(-1, *next_rgb_obs.shape[2:])
        return rgb_obs, next_rgb_obs, metrics

    def encode(self, rgb_obs, next_rgb_obs):
        metrics = dict()

        # Update the encoder with auxiliary objective if we need to.
        metrics.update(self.update_encoder_rep(rgb_obs.float()))

        # Extract the features from the encoder
        multi_view_rgb_feats = self.encoder(rgb_obs.float())

        with torch.no_grad():
            next_multi_view_rgb_feats = self.encoder(next_rgb_obs.float())

        return metrics, multi_view_rgb_feats, next_multi_view_rgb_feats

    def multi_view_fusion(self, rgb_obs, multi_view_feats, next_multi_view_feats):
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

        with torch.no_grad():
            next_fused_view_feats = self.view_fusion(next_multi_view_feats)

        return metrics, fused_view_feats, next_fused_view_feats

    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        (
            metrics,
            batch,
            action,
            reward,
            discount,
            terminal,
            truncated,
            bootstrap,
            demos,
            time_obs,
            next_time_obs,
            loss_coeff,
        ) = self.extract_batch(replay_iter)

        low_dim_obs = next_low_dim_obs = None
        fused_view_feats = next_fused_view_feats = None
        if self.low_dim_size > 0:
            low_dim_obs, next_low_dim_obs = self.extract_low_dim_state(batch)

        if self.use_pixels:
            rgb_obs, next_rgb_obs, rgb_metrics = self.extract_pixels(batch)
            metrics.update(rgb_metrics)
            enc_metrics, rgb_feats, next_rgb_feats = self.encode(rgb_obs, next_rgb_obs)
            metrics.update(enc_metrics)
            (
                fusion_metrics,
                fused_view_feats,
                next_fused_view_feats,
            ) = self.multi_view_fusion(rgb_obs, rgb_feats, next_rgb_feats)
            metrics.update(fusion_metrics)
            if not self.frame_stack_on_channel:
                fused_view_feats = fused_view_feats.view(
                    -1, self.time_dim, *fused_view_feats.shape[1:]
                )
                next_fused_view_feats = next_fused_view_feats.view(
                    -1, self.time_dim, *next_fused_view_feats.shape[1:]
                )

        metrics.update(
            self.update_critic(
                low_dim_obs,
                fused_view_feats,
                action,
                reward,
                discount,
                bootstrap,
                next_low_dim_obs,
                next_fused_view_feats,
                step,
                time_obs,
                next_time_obs,
                loss_coeff,
                False,
            )
        )

        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            replay_buffer.set_priority(
                indices=batch["indices"].cpu().detach().numpy(),
                priorities=self._td_error**self.replay_alpha,
            )

        if self.intrinsic_reward_module is not None:
            intrinsic_rewards = self.intrinsic_reward_module.compute_irs(batch, step)
            self.intrinsic_reward_module.update(batch)
            metrics.update(
                self.update_critic(
                    low_dim_obs,
                    fused_view_feats.detach() if fused_view_feats is not None else None,
                    action,
                    intrinsic_rewards,
                    discount,
                    bootstrap,
                    next_low_dim_obs,
                    next_fused_view_feats.detach()
                    if next_fused_view_feats is not None
                    else None,
                    step,
                    time_obs,
                    next_time_obs,
                    loss_coeff,
                    True,
                )
            )
            utils.soft_update_params(
                self.intr_critic, self.intr_critic_target, self.critic_target_tau
            )

        if fused_view_feats is not None:
            fused_view_feats = fused_view_feats.detach()
        metrics.update(
            self.update_actor(
                low_dim_obs, fused_view_feats, action, step, time_obs, demos, loss_coeff
            )
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def calculate_target_q(
        self,
        step,
        next_low_dim_obs,
        next_fused_view_feats,
        next_time_obs,
        reward,
        discount,
        bootstrap,
        updating_intrinsic_critic,
    ):
        critic_target = (
            self.intr_critic_target if updating_intrinsic_critic else self.critic_target
        )
        with torch.no_grad():
            dist = self.actor(next_low_dim_obs, next_fused_view_feats)
            next_action = dist.sample()
            target_qs = critic_target(
                next_low_dim_obs, next_fused_view_feats, next_action, next_time_obs
            )
            if self.distributional_critic:
                target_qs = critic_target.from_dist(target_qs)

            min_q = target_qs.min(-1, keepdim=True)[0]
            target_q = reward + bootstrap * discount * min_q
            if self.distributional_critic:
                return critic_target.to_dist(target_q)
            return target_q

    def reset(self, step: int, agents_to_reset: list[int]):
        for aid in agents_to_reset:
            self.actor.reset(aid)
            self.critic.reset(aid)
            if self.intr_critic is not None:
                self.intr_critic.reset(aid)

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value
        self.actor.set_eval_env_running(value)
        self.critic.set_eval_env_running(value)
        if self.intr_critic is not None:
            self.intr_critic.set_eval_env_running(value)

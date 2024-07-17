from abc import ABC
from copy import deepcopy
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from robobase import utils
from robobase.method.core import OffPolicyMethod
from robobase.models.fusion import FusionModule
from robobase.models.encoder import EncoderModule
from robobase.models.fully_connected import (
    FullyConnectedModule,
    RNNFullyConnectedModule,
)
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
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
    RandomShiftsAug,
)
import torch.nn.functional as F


class Critic(nn.Module):
    """Critic for value-based agent that discretizes action space into uniform bins.

    This class is responsible for computing Q-values with discretized bins.
    A core component is encoding/decoding actions, which allows for maintaining other
    parts of code as continuous, without introducing a wrapper for discretized actions
    """

    def __init__(
        self,
        actor_dim: int,
        bins: int,
        advantage_model: FullyConnectedModule,
        value_model: Optional[FullyConnectedModule] = None,
    ):
        """Initializes a critic

        Args:
            actor_dim: Dimension of action space.
            bins: Number of bins used for discretizing continuous action space
            advantage_model: Class to use for advantage network
            value_model: (Optional) Class to use for value network when using Dueling
        """
        super().__init__()
        self.adv = advantage_model
        self.value = value_model
        self.use_dueling = value_model is not None
        self.bins = bins

        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )

        self.apply(utils.weight_init)
        self.adv.initialize_output_layer(utils.uniform_weight_init(0.0))
        if self.use_dueling:
            self.value.initialize_output_layer(utils.uniform_weight_init(0.0))

    def forward(self, low_dim_obs, fused_view_feats, action, time_obs):
        """Compute Q-values for each bins.

        The network first computes Q-values for all bins ([B, D, bins])
        Then gathers Q-values for bins specified by actions ([B, D])

        Args:
            low_dim_obs: low-dimensional observations
            fused_view_feats: rgb multi-view fused observations
            action: Index of bins to evaluate Q-values
            time_obs: time observations

        Returns:
            Q-value corresponding to bins specified by input actions.
        """
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        if time_obs is not None:
            net_ins["time_obs"] = time_obs

        advs = self.adv(net_ins)
        if self.use_dueling:
            values = self.value(net_ins)
            qs = values + advs - advs.mean(-1, keepdim=True)
        else:
            qs = advs  # [B, D, bins]

        discrete_action = self.encode_action(action).long()  # [B, D]
        qs_a = torch.gather(qs, dim=-1, index=discrete_action.unsqueeze(-1))[..., 0]
        return qs_a, qs

    def get_action(self, low_dim_obs, fused_view_feats, time_obs, intr_critic=None):
        """Outputs actions.

        The network first computes Q-values for all bins ([B, D, bins])
        Then finds the argmax indices of maximum q-value for each dimension.
        Then it computes the continuous action for each bins

        If optional `intr_critic` is specified, the network also computes
        intrinsic Q-values for each bins, and find the argmax indices of
        maximum (q_values + intrinsic q_values) for each dimension

        Args:
            low_dim_obs: low-dimensional observations
            fused_view_feats: rgb multi-view fused observations
            time_obs: time observations
            intr_critic: (optional) intrinsic critic network

        Returns:
            Continuous actions of shape [B, D]
        """
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
            bs = low_dim_obs.shape[0]
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
            bs = fused_view_feats.shape[0]
        if time_obs is not None:
            net_ins["time_obs"] = time_obs
            bs = time_obs.shape[0]

        low = self.initial_low.repeat(bs, 1).detach()
        high = self.initial_high.repeat(bs, 1).detach()

        advs = self.adv(net_ins)
        if self.use_dueling:
            values = self.value(net_ins)
            qs = values + advs - advs.mean(-1, keepdim=True)
        else:
            qs = advs  # [B, D, bins]

        if intr_critic is not None:
            # Get action with with intrinsic critic
            intr_advs = intr_critic.adv(net_ins)
            if self.use_dueling:
                intr_values = intr_critic.value(net_ins)
                intr_qs = intr_values + intr_advs - intr_advs.mean(-1, keepdim=True)
            else:
                intr_qs = intr_advs
            qs += intr_qs

        argmax_q = random_action_if_within_delta(qs)
        if argmax_q is None:
            argmax_q = qs.max(-1)[1]
        # Zoom-in
        low, high = zoom_in(low, high, argmax_q, self.bins)
        continuous_action = (high + low) / 2.0
        return continuous_action

    def encode_action(self, continuous_action):
        """Encodes continuous actions to discrete actions

        Args:
            continuous action: continuous action of shape [B, D]
        Returns:
            Discrete actions of shape [B, D], consisting of bin indices
            e.g.) [[1, 4, 3, 2, 2], ..., [0, 2, 2, 1, 3]]
        """
        return encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            1,
            self.bins,
        )[..., 0, :]

    def decode_action(self, discrete_action):
        """Decodes discrete actions to continuous actions

        Args:
            discrete action: discrete action of shape [B, D]
        Returns:
            Continuous actions of shape [B, D]
        """
        # decode_action expects [B, 1, D], a bit hacky
        discrete_action = discrete_action[:, None, :]
        x = decode_action(
            discrete_action, self.initial_low, self.initial_high, 1, self.bins
        )
        return x

    def encode_decode_action(self, action: torch.Tensor):
        return self.decode_action(self.encode_action(action))


class ValueBased(OffPolicyMethod, ABC):
    """Value-based agent that discretizes action space into uniform bins.

    Note that we assume we only use this agent for *continuous* control, so that
    it's not possible to use this class for discrete control tasks (e.g., Atari)
    """

    def __init__(
        self,
        num_explore_steps: int,
        critic_lr: float,
        view_fusion_lr: float,
        encoder_lr: float,
        weight_decay: float,
        num_critics: int,
        critic_target_tau: float,
        critic_grad_clip: Optional[float],
        bins: int,
        always_bootstrap: bool,
        stddev_schedule: str,
        use_dueling: bool,
        bc_lambda: float,
        bc_margin: float,
        use_target_network_for_rollout: bool,
        num_update_steps: int,
        use_augmentation: bool,
        use_torch_compile: bool,
        advantage_model: Optional[FullyConnectedModule],
        value_model: Optional[FullyConnectedModule],
        encoder_model: Optional[EncoderModule],
        view_fusion_model: Optional[FusionModule],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert (
            num_critics == 1
        ), f"num_critics > 1 is not supported, but set to {num_critics}"
        self.num_explore_steps = num_explore_steps
        self.critic_lr = critic_lr
        self.view_fusion_lr = view_fusion_lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.critic_target_tau = critic_target_tau
        self.bins = bins
        self.always_bootstrap = always_bootstrap
        self.stddev_schedule = stddev_schedule
        self.use_dueling = use_dueling
        self.bc_lambda = bc_lambda
        self.bc_margin = bc_margin
        self.use_target_network_for_rollout = use_target_network_for_rollout
        self.num_update_steps = num_update_steps
        self.aug = RandomShiftsAug(pad=4) if use_augmentation else lambda x: x
        self.use_torch_compile = use_torch_compile
        self.critic_grad_clip = critic_grad_clip
        self.advantage_model = advantage_model
        self.value_model = value_model
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        self.use_pixels = len(self.rgb_spaces) > 0
        self.use_multicam_fusion = len(self.rgb_spaces) > 1
        # T should be same across all obs
        self.time_dim = list(self.observation_space.values())[0].shape[0]
        self.critic = self.encoder = self.view_fusion = None
        self.build_encoder()
        self.build_view_fusion()
        self.critic, self.critic_target, self.critic_opt = self.build_critic()
        self.intr_critic = self.intr_critic_target = self.intr_critic_opt = None
        if self.intrinsic_reward_module:
            (
                self.intr_critic,
                self.intr_critic_target,
                self.intr_critic_opt,
            ) = self.build_critic()

    def build_critic(self):
        """Build critic for ValueBased agent

        When using dueling network, a separate value_model is initialized too.

        Returns:
            critic, target critic, and optimizer for critic
        """
        critic_cls = Critic
        actor_dim = np.prod(self.action_space.shape)
        input_shapes = self.get_fully_connected_inputs()
        advantage_model = self.advantage_model(
            input_shapes=input_shapes,
            output_shape=(actor_dim, self.bins),
            num_envs=self.num_train_envs + 1,
        )
        if self.use_dueling:
            value_model = self.value_model(
                input_shapes=input_shapes,
                output_shape=(actor_dim, 1),
                num_envs=self.num_train_envs + 1,
            )
        else:
            value_model = None
        critic = critic_cls(
            actor_dim,
            self.bins,
            advantage_model,
            value_model,
        ).to(self.device)
        critic_target = deepcopy(critic)
        critic_target.load_state_dict(critic.state_dict())
        critic_opt = torch.optim.AdamW(
            critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay
        )
        critic_target.eval()
        if self.use_torch_compile:
            critic = torch.compile(critic)
            critic_target = torch.compile(critic_target)
        return critic, critic_target, critic_opt

    def build_encoder(self):
        """Build RGB encoder for ValueBased agent"""
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
            if self.use_torch_compile:
                self.encoder = torch.compile(self.encoder)

    def build_view_fusion(self):
        """Build view_fusion model for ValueBased agent"""
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
                if self.use_torch_compile:
                    self.view_fusion = torch.compile(self.view_fusion)
            self.rgb_latent_size = self.view_fusion.output_shape[-1]
        else:
            self.view_fusion = lambda x: x[:, 0]
            self.rgb_latent_size = self.encoder.output_shape[-1]

    def get_fully_connected_inputs(self) -> dict[str, tuple]:
        """Get input_sizes for FullyConnectedModules"""
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
        """Try to update encoder representations if it has `calculcate_loss`

        Args:
            rgb_obs: RGB observations
        Returns:
            empty dictionary if the encoder has no particular `calculate_loss`
            or loss for updating the encoder if it has one
        """
        loss = self.encoder.calculate_loss(rgb_obs)
        if loss is None:
            return {}
        self.encoder.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_opt.step()
        return {"encoder_rep_loss": loss.item()}

    def update_view_fusion_rep(self, rgb_feats):
        """Try to update view_fusion model if it has `calculcate_loss`

        Args:
            rgb_feats: RGB features encoded by the encoder
        Returns:
            empty dictionary if view_fusion model has no particular `calculate_loss`
            or loss for updating the view_fusion model if it has one
        """
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
            std = self.get_std(step)
            with torch.no_grad():
                return self._act(observations, eval_mode, std)

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

    def _act_extract_time_obs(self, observations: dict[str, torch.Tensor]):
        time_obs = extract_from_spec(observations, "time", missing_ok=True)
        if time_obs is not None:
            time_obs = time_obs.float()[:, -1]
        return time_obs

    def _act(
        self,
        observations: dict[str, torch.Tensor],
        eval_mode: bool,
        std: float,
    ):
        low_dim_obs = fused_rgb_feats = time_obs = None
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
        if self.time_obs_size > 0:
            time_obs = self._act_extract_time_obs(observations)
        if self.use_target_network_for_rollout:
            critic = self.critic_target
            intr_critic = self.intr_critic_target
        else:
            critic = self.critic
            intr_critic = self.intr_critic
        action = critic.get_action(low_dim_obs, fused_rgb_feats, time_obs, intr_critic)
        std = torch.ones_like(action) * std
        dist = utils.TruncatedNormal(action, std)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()

        action = self.critic.encode_decode_action(action)
        # Unflatten to include action_sequence dimension
        action = action.view(*action.shape[:-1], *self.action_space.shape)
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
        next_action,
        time_obs,
        next_time_obs,
        loss_coeff,
        demos,
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
        target_qs_a = self.calculate_target_q(
            next_low_dim_obs,
            next_fused_view_feats,
            next_action,
            next_time_obs,
            reward,
            discount,
            bootstrap,
            updating_intrinsic_critic,
        )

        qs_a, qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
        q_critic_loss = F.mse_loss(qs_a, target_qs_a, reduction="none").mean(-1)
        critic_loss = q_critic_loss * loss_coeff

        # Compute priority
        new_pri = torch.sqrt(q_critic_loss + 1e-10)
        self._td_error = (new_pri / torch.max(new_pri)).cpu().detach().numpy()
        critic_loss = torch.mean(critic_loss)

        if self.logging:
            metrics[f"{lp}critic_target_q"] = target_qs_a.mean().item()
            metrics[f"{lp}critic_q"] = qs_a.mean().item()
            metrics[f"{lp}critic_loss"] = critic_loss.item()

        if self.bc_lambda > 0.0 and not updating_intrinsic_critic:
            # No BC loss when updating intrinsic critic
            if demos is not None and torch.sum(demos) > 0:
                margin_loss = torch.clamp(
                    self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
                ).mean([-3, -2, -1])
                margin_loss = (margin_loss * demos).sum() / demos.sum()
            else:
                margin_loss = torch.tensor(0.0, device=demos.device)
            critic_loss = critic_loss + self.bc_lambda * margin_loss
            if self.logging:
                metrics[f"{lp}margin_loss"] = margin_loss.item()

        # optimize encoder and critic
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.critic_grad_clip:
            critic_norm = nn.utils.clip_grad_norm_(
                critic.parameters(), self.critic_grad_clip
            )
            if self.logging:
                metrics[f"{lp}critic_norm"] = critic_norm.item()
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
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}
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

        # Flatten action sequence dimension
        action = action.flatten(-2)

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

        # Augmentation
        b, v, c, h, w = rgb_obs.shape
        rgb_obs = self.aug(rgb_obs.float().view(b * v, c, h, w)).view(b, v, c, h, w)
        next_rgb_obs = self.aug(next_rgb_obs.float().view(b * v, c, h, w)).view(
            b, v, c, h, w
        )
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
        if step != 0:
            num_update_steps = self.num_update_steps
        else:
            num_update_steps = 1  # pre-training when step == 0
        for _ in range(num_update_steps):
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
                rgb_obs, next_rgb_obs, ext_metrics = self.extract_pixels(batch)
                metrics.update(ext_metrics)
                enc_metrics, rgb_feats, next_rgb_feats = self.encode(
                    rgb_obs, next_rgb_obs
                )
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

            with torch.no_grad():
                next_action = self.critic.get_action(
                    next_low_dim_obs,
                    next_fused_view_feats,
                    next_time_obs,
                    self.intr_critic,
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
                    next_action,
                    time_obs,
                    next_time_obs,
                    loss_coeff,
                    demos,
                    False,
                )
            )

            if isinstance(replay_buffer, PrioritizedReplayBuffer):
                replay_buffer.set_priority(
                    indices=batch["indices"].cpu().detach().numpy(),
                    priorities=self._td_error**self.replay_alpha,
                )

            if self.intrinsic_reward_module is not None:
                intrinsic_rewards = self.intrinsic_reward_module.compute_irs(
                    batch, step
                )
                self.intrinsic_reward_module.update(batch)
                metrics.update(
                    self.update_critic(
                        low_dim_obs,
                        fused_view_feats.detach()
                        if fused_view_feats is not None
                        else None,
                        action,
                        intrinsic_rewards,
                        discount,
                        bootstrap,
                        next_low_dim_obs,
                        next_fused_view_feats.detach()
                        if next_fused_view_feats is not None
                        else None,
                        next_action,
                        time_obs,
                        next_time_obs,
                        loss_coeff,
                        None,
                        True,
                    )
                )
                utils.soft_update_params(
                    self.intr_critic, self.intr_critic_target, self.critic_target_tau
                )

            # update critic target
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )

        return metrics

    def calculate_target_q(
        self,
        next_low_dim_obs,
        next_fused_view_feats,
        next_action,
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
            target_qs_a, _ = critic_target(
                next_low_dim_obs, next_fused_view_feats, next_action, next_time_obs
            )
            target_q = reward + bootstrap * discount * target_qs_a
            return target_q

    def get_std(self, step):
        return utils.schedule(self.stddev_schedule, step)

    def reset(self, step: int, agents_to_reset: list[int]):
        for aid in agents_to_reset:
            if isinstance(self.advantage_model, RNNFullyConnectedModule):
                self.advantage_model.reset(aid)
            if self.use_dueling:
                if isinstance(self.value_model, RNNFullyConnectedModule):
                    self.value_model.reset(aid)

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value
        if isinstance(self.advantage_model, RNNFullyConnectedModule):
            self.advantage_model.eval_env_running = value
        if self.use_dueling:
            if isinstance(self.value_model, RNNFullyConnectedModule):
                self.value_model.eval_env_running = value

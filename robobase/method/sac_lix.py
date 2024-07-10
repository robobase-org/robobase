from distutils.dist import Distribution

import numpy as np
import torch
import torch.nn as nn
from robobase.method.utils import extract_many_from_spec

from robobase.models.lix_utils import analysis_optimizers

from robobase import utils
from robobase.method.actor_critic import ActorCritic, Actor
from robobase.models.lix_utils.analysis_modules import LIXModule

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SACActor(Actor):
    def forward(self, low_dim_obs, fused_view_feats) -> Distribution:
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        mu, log_std = self.actor_model(net_ins).chunk(2, -1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        dist = utils.SquashedNormal(mu, std)
        return dist

    def logprob(self, dist):
        tanh_a = dist.rsample()
        log_prob = dist.log_prob(tanh_a)
        # Enforcing Action Bound
        log_prob = log_prob.sum(-1)
        return tanh_a, log_prob


class SACLix(ActorCritic):
    """Implementation of Soft Actor-Critic (SAC) with Local sIgnal miXing (LIX) layer.

    Haarnoja, Tuomas, et al. "Soft actor-critic:
    Off-policy maximum entropy deep reinforcement learning with a stochastic actor."
    """

    def __init__(self, alpha_lr: float, init_temperature: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_temperature = init_temperature
        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_space.shape).to(self.device)
        ).item()
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(init_temperature), device=self.device)
        )
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

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
            if not isinstance(self.encoder, LIXModule):
                raise ValueError("Encoder must be of type ALIXModule.")
            self.encoder.to(self.device)
            self.encoder_opt = (
                analysis_optimizers.custom_parameterized_aug_optimizer_builder(
                    encoder_lr=self.encoder_lr, lr=2e-3, betas=[0.5, 0.999]
                )(self.encoder)
            )

    def build_actor(self):
        self.actor_model = self.actor_model(
            input_shapes=self.get_fully_connected_inputs(),
            output_shape=self.action_space.shape[-1] * 2,
            num_envs=self.num_train_envs + self.num_eval_envs,
        )
        self.actor = SACActor(self.actor_model).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

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
            next_action, log_prob = self.actor.logprob(dist)
            log_prob = log_prob.mean(-1, keepdim=True)  # account for action sequence
            target_qs = critic_target(
                next_low_dim_obs, next_fused_view_feats, next_action, next_time_obs
            )
            if self.distributional_critic:
                target_qs = critic_target.from_dist(target_qs)
            min_q = (
                target_qs.min(-1, keepdim=True)[0]
                - self.log_alpha.exp().detach() * log_prob
            )
            target_q = reward + bootstrap * discount * min_q
            if self.distributional_critic:
                return critic_target.to_dist(target_q)
            return target_q

    def _compute_actor_loss(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        time_obs,
        loss_coeff,
        critic,
        log_prob,
    ):
        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
        if self.distributional_critic:
            qs = critic.from_dist(qs)
        min_q = qs.min(-1, keepdim=True)[0]
        return (
            ((self.log_alpha.exp().detach() * log_prob) - min_q)
            * loss_coeff.unsqueeze(1)
        ).mean()

    def update_actor(
        self, low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
    ):
        metrics = dict()

        dist = self.actor(low_dim_obs, fused_view_feats)
        action, log_prob = self.actor.logprob(dist)

        base_actor_loss = self._compute_actor_loss(
            low_dim_obs,
            fused_view_feats,
            action,
            time_obs,
            loss_coeff,
            self.critic,
            log_prob,
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
                log_prob,
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

        self.a_optimizer.zero_grad()
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.a_optimizer.step()
        alpha = self.log_alpha.exp().item()
        if self.logging:
            metrics["alpha"] = alpha
            metrics["alpha_loss"] = alpha_loss
            metrics["mean_act"] = dist.mean.mean().item()
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
        return metrics

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from robobase.method.drqv2 import DrQV2
from robobase.method.actor_critic import Critic, DistributionalCritic


class IQLDrQV2(DrQV2):
    def __init__(
        self,
        expectile,
        actor_loss_type="ddpg",
        awr_temperature=3.0,
        *args,
        **kwargs,
    ):
        """Init function for Implicit Q Learning (https://arxiv.org/abs/2110.06169).

        IQL is an offline RL algorithm that aims to avoid querying out-of-distribution
        actions during training. Specifically, it tries to avoid sampling actions
        a \sim \pi(a|s) during policy evaluation, which might cause over-estimation of
        Q values with the dynamic programming of Bellman's optimality equation
        Q(s,a) = r(s,a) + \gamma \max_a' Q(s',a'). Instead, IQL learns an value function
        V(s) with expectile regression that approximates the maximum Q values, and
        performs policy evaluation with Q(s, a) = r(s, a) + \gamma V(s').

        During policy improvement, DDPG-style loss trains a policy by maximizing the
        Q function, i.e., argmax_\pi E_{a~\pi}[Q(s, a)]. We also provide the Advantage
        Weighted Regression (AWR, https://arxiv.org/abs/1910.00177) used in the original
        IQL paper. Unlike DDPG, AWR performs behavior cloning weighted by an exponential
        of the advantage value A(s, a) = Q(s, a) - V(s). AWR objective is derived
        in closed form by solving an one-step improved policy under constraints. We
        would recommend using DDPG-style actor loss for off-policy RL with IQL,
        and AWR can be considered when expert demos are available.

        Args:
            expectile: Expectile ratio for expectile regression in IQL.
            actor_loss_type: Actor loss type in IQL used to train the actor.
                             It could be ddpg or awr.
            awr_temperature: Temperature for awr in IQL.
        """
        super().__init__(*args, **kwargs)
        self.expectile = expectile
        self.actor_loss_type = actor_loss_type
        self.awr_temperature = awr_temperature

        # Overwrite the critic_opt defined in super class
        self.value_fn, self.critic_opt = self._build_value(self.critic)
        self.intr_value_fn = self.intr_critic_opt = None
        if self.intrinsic_reward_module:
            self.intr_value_fn, self.intr_critic_opt = self._build_value(
                self.intr_critic
            )

    def _build_value(self, critic):
        if self.distributional_critic:
            value_fn = DistributionalCritic(
                self.distributional_critic_limit,
                self.distributional_critic_atoms,
                self.distributional_critic_transform,
                deepcopy(self.critic_model),
                1,
            ).to(self.device)
        else:
            value_fn = Critic(deepcopy(self.critic.qs[0]), 1).to(self.device)
        critic_opt = torch.optim.AdamW(
            list(critic.parameters()) + list(value_fn.parameters()),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
        )
        return value_fn, critic_opt

    def _calc_adv(self, low_dim_obs, fused_view_feats, act, time_obs, critic, value_fn):
        qs = critic(low_dim_obs, fused_view_feats, act, time_obs)
        dummy_actions = torch.zeros(
            [low_dim_obs.size(0), *self.actor.actor_model.output_shape],
            dtype=low_dim_obs.dtype,
            device=low_dim_obs.device,
        )
        v = value_fn(low_dim_obs, fused_view_feats, dummy_actions, time_obs)
        if self.distributional_critic:
            qs = critic.from_dist(qs)
            v = value_fn.from_dist(v)
        min_q = qs.min(-1, keepdim=True)[0]
        return torch.exp((min_q - v) * self.awr_temperature).clip(max=100.0)

    def _awr_loss(
        self, low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
    ):
        metrics = dict()

        std = self.get_std(step)
        dist = self.actor(low_dim_obs, fused_view_feats, std)
        log_prob = dist.log_prob(act).sum(-1, keepdim=True)

        with torch.no_grad():
            adv = self._calc_adv(
                low_dim_obs, fused_view_feats, act, time_obs, self.critic, self.value_fn
            )
            intrinsic_actor_loss = 0
            if self.intrinsic_reward_module is not None:
                intr_adv = self._calc_adv(
                    low_dim_obs,
                    fused_view_feats,
                    act,
                    time_obs,
                    self.intr_critic,
                    self.intr_value_fn,
                )
                intrinsic_actor_loss = -(intr_adv * log_prob).mean()

        actor_loss = -(adv * log_prob).mean() + intrinsic_actor_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()

        if self.logging:
            metrics["actor_loss"] = actor_loss.item()
            metrics["adv"] = adv.mean().item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def update_actor(
        self, low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
    ):
        if self.actor_loss_type == "ddpg":
            fn = super().update_actor
        elif self.actor_loss_type == "awr":
            fn = self._awr_loss
        else:
            raise NotImplementedError("Unimplemented actor loss type for IQL.")

        return fn(low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff)

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
            critic, critic_opt, value_fn = (
                self.intr_critic,
                self.intr_critic_opt,
                self.intr_value_fn,
            )
        else:
            critic, critic_opt, value_fn = (
                self.critic,
                self.critic_opt,
                self.value_fn,
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
        target_qs = target_qs.repeat(1, self.num_critics)

        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
        # Compute priority
        q_critic_loss = F.mse_loss(qs, target_qs.detach(), reduction="none").mean(
            -1, keepdim=True
        )

        dummy_action = torch.zeros_like(action)
        iql_q = value_fn(low_dim_obs, fused_view_feats, dummy_action, time_obs)
        min_q = qs.min(-1, keepdim=True)[0]
        if self.distributional_critic:
            iql_q = critic.from_dist(iql_q)
            min_q = critic.from_dist(min_q)

        vf_err = min_q.detach() - iql_q
        vf_weight = torch.where(vf_err > 0, self.expectile, (1 - self.expectile))
        vf_loss = (vf_weight * (vf_err**2)).mean()
        critic_loss = (q_critic_loss + vf_loss) * loss_coeff.unsqueeze(1)

        new_pri = torch.sqrt(q_critic_loss + 1e-10)
        self._td_error = (new_pri / torch.max(new_pri)).cpu().detach().numpy()
        critic_loss = torch.mean(critic_loss)

        if self.logging:
            metrics[f"{lp}critic_target_q"] = target_qs.mean().item()
            for i in range(1, self.num_critics):
                metrics[f"{lp}critic_q{i + 1}"] = qs[..., i].mean().item()
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
        value_fn = self.intr_value_fn if updating_intrinsic_critic else self.value_fn
        with torch.no_grad():
            dummy_actions = torch.zeros(
                [reward.size(0), *self.actor.actor_model.output_shape],
                dtype=reward.dtype,
                device=reward.device,
            )
            v = value_fn(
                next_low_dim_obs, next_fused_view_feats, dummy_actions, next_time_obs
            )
            if self.distributional_critic:
                v = value_fn.from_dist(v)

            target_q = reward + bootstrap * discount * v
            return target_q

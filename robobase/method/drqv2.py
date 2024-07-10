import torch
import torch.nn as nn
from torch.distributions import Distribution

from robobase import utils
from robobase.method.actor_critic import Actor as BaseActor
from robobase.method.actor_critic import ActorCritic
from robobase.method.utils import RandomShiftsAug


class Actor(BaseActor):
    def forward(self, low_dim_obs, fused_view_feats, std) -> Distribution:
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        mu = self.actor_model(net_ins)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class DrQV2(ActorCritic):
    def __init__(self, stddev_schedule, stddev_clip, use_augmentation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.aug = RandomShiftsAug(pad=4) if use_augmentation else lambda x: x

    def build_actor(self):
        input_shapes = self.get_fully_connected_inputs()
        if "time_obs" in input_shapes:
            # We don't use time_obs for actor
            input_shapes.pop("time_obs")
        self.actor_model = self.actor_model(
            input_shapes=input_shapes,
            output_shape=self.action_space.shape[-1],
            num_envs=self.num_train_envs + self.num_eval_envs,
        )
        self.actor = Actor(self.actor_model).to(self.device)
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay
        )

    def _act(
        self,
        observations: dict[str, torch.Tensor],
        eval_mode: bool,
        std: float,
    ):
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
        dist = self.actor(low_dim_obs, fused_rgb_feats, std)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        return action

    def act(self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool):
        if step < self.num_explore_steps and not eval_mode:
            return self.random_explore_action
        else:
            std = self.get_std(step)
            with torch.no_grad():
                return self._act(observations, eval_mode, std)

    def update_actor(
        self, low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
    ):
        metrics = dict()

        std = self.get_std(step)
        dist = self.actor(low_dim_obs, fused_view_feats, std)
        action = dist.sample(clip=self.stddev_clip)
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

    def extract_pixels(self, batch: dict[str, torch.Tensor]):
        rgb_obs, next_rgb_obs, metrics = super().extract_pixels(batch)
        b, v, c, h, w = rgb_obs.shape
        rgb_obs = self.aug(rgb_obs.float().view(b * v, c, h, w)).view(b, v, c, h, w)
        next_rgb_obs = self.aug(next_rgb_obs.float().view(b * v, c, h, w)).view(
            b, v, c, h, w
        )
        return rgb_obs, next_rgb_obs, metrics

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
            std = self.get_std(step)
            dist = self.actor(next_low_dim_obs, next_fused_view_feats, std)
            next_action = dist.sample(clip=self.stddev_clip)
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

    def get_std(self, step):
        return utils.schedule(self.stddev_schedule, step)

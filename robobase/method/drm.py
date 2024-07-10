from copy import deepcopy
from collections import deque

import numpy as np
import torch
from torch import nn

from robobase import utils
from robobase.method.drqv2 import DrQV2
from robobase.models.fully_connected import FullyConnectedModule


class DrMValue(nn.Module):
    def __init__(
        self,
        value_model: FullyConnectedModule,
    ):
        super().__init__()
        self.value_net = value_model
        self.apply(utils.weight_init)

    def forward(self, low_dim_obs, fused_view_feats, time_obs):
        net_ins = dict()
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        if time_obs is not None:
            net_ins["time_obs"] = time_obs
        return self.value_net(net_ins)

    def reset(self, env_index: int):
        self.value_net.reset(env_index)

    def set_eval_env_running(self, value: bool):
        self.value_net.eval_env_running = value


class DormantRatioTracker:
    def __init__(self, actor: nn.Module, t_dormant_ratio: float):
        self.value = 1
        self.eval_mode = False

        self._actor = actor
        self._t_dormant_ratio = t_dormant_ratio
        self._dormant_count, self._total_count = 0, 0

        self.register_dormant_hooks(self._actor.modules())
        self._actor.register_forward_hook(self._calculate_net_dormant_ratio)

    def register_dormant_hooks(self, layers):
        queue = deque(maxlen=2)  # Queue to keep track of the last two layers
        for layer in layers:
            if isinstance(layer, nn.ReLU):
                # Check if the last layer is nn.Linear
                # or the sequence is [nn.Linear, nn.Identity]
                error = True
                if isinstance(queue[-1], nn.Linear):
                    error = False
                if len(queue) == 2 and (
                    isinstance(queue[-2], nn.Linear)
                    and isinstance(queue[-1], nn.Identity)
                ):
                    error = False
                if error:
                    raise ValueError(
                        "This DRM implementation relies on the usage of "
                        "nn.Linear layers followed by nn.ReLU layers. "
                        "Please ensure the correct structure of the Actor."
                    )
                layer.register_forward_hook(self._calculate_layer_dormant_ratio)
            queue.append(layer)

    def _calculate_layer_dormant_ratio(self, module, inputs, outputs):
        # outputs: [B, ..., D]
        # Convert outputs to [BB, D] by flattening
        flat_outputs = outputs.flatten(0, -2)
        # Compute score for each neuron -> [D,]
        score = torch.mean(torch.abs(flat_outputs), dim=0)
        # Normalize the score
        score = score / (torch.mean(score) + 1e-9)
        self._dormant_count += torch.sum(score <= self._t_dormant_ratio).item()
        self._total_count += score.numel()

    def _calculate_net_dormant_ratio(self, module, inputs, outputs):
        if self._actor.training and not self.eval_mode:
            self.value = self._dormant_count / self._total_count
        self.eval_mode = False
        self._dormant_count, self._total_count = 0, 0


class DrM(DrQV2):
    """Implementation of Dormant Ratio Minimization (DRM), using DRQ as base.
    https://arxiv.org/abs/2310.19668
    """

    def __init__(
        self,
        value_model: FullyConnectedModule,
        value_lr: float,
        t_dormant_ratio: float,
        dormant_ratio_threshold: float,
        awaken_exploration_temperature: float,
        exploitation_temperature: float,
        exploitation_lam_max: float,
        exploitation_expectile: float,
        perturbation_frames: int,
        minimum_perturb_factor: float,
        maximum_perturb_factor: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.value_model = value_model
        self.t_dormant_ratio = t_dormant_ratio
        self.dormant_ratio_threshold = dormant_ratio_threshold
        self.awaken_exploration_temperature = awaken_exploration_temperature
        self.exploitation_temperature = exploitation_temperature
        self.exploitation_lam_max = exploitation_lam_max
        self.exploitation_expectile = exploitation_expectile
        self.perturbation_frames = perturbation_frames
        self.minimum_perturb_factor = minimum_perturb_factor
        self.maximum_perturb_factor = maximum_perturb_factor

        # Dormant ratio calculation
        self.dormant_ratio_tracker = DormantRatioTracker(self.actor, t_dormant_ratio)
        # DrM value function
        value_model = self.value_model(
            input_shapes=self.get_fully_connected_inputs(),
            num_envs=self.num_train_envs + self.num_eval_envs,
        )
        self.value = DrMValue(value_model).to(self.device)
        self.value_opt = torch.optim.AdamW(
            self.value.parameters(), lr=value_lr, weight_decay=self.weight_decay
        )
        self.value.train()
        self.intr_value = self.intr_value_opt = None
        if self.intrinsic_reward_module:
            self.intr_value = DrMValue(value_model).to(self.device)
            self.intr_value_opt = torch.optim.AdamW(
                self.intr_value.parameters(),
                lr=value_lr,
                weight_decay=self.weight_decay,
            )
            self.intr_value.train()
        self.awaken_step = None
        self.reset_step = 0

    def get_std(self, step):
        stddev = self._drm_stddev(
            self.dormant_ratio_tracker.value,
            self.dormant_ratio_threshold,
            self.awaken_exploration_temperature,
            True,
        )
        if self.awaken_step is not None:
            stddev = max(
                stddev, utils.schedule(self.stddev_schedule, step - self.awaken_step)
            )
        return stddev

    def update_actor(
        self, low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
    ):
        metrics = super().update_actor(
            low_dim_obs, fused_view_feats, act, step, time_obs, demos, loss_coeff
        )

        if self.logging:
            metrics["dormant_ratio"] = self.dormant_ratio_tracker.value

        # Optimize DrM value function
        metrics.update(
            self.update_value(low_dim_obs, fused_view_feats, act, time_obs, False)
        )

        # Update Drm awaken step
        self.update_awaken_step(step)

        if self.intrinsic_reward_module is not None:
            metrics.update(
                self.update_value(
                    low_dim_obs,
                    fused_view_feats.detach() if fused_view_feats is not None else None,
                    act,
                    time_obs,
                    True,
                )
            )

        # DrM perturbation
        if step - self.reset_step >= self.perturbation_frames:
            alpha = np.clip(
                1 - 2 * self.dormant_ratio_tracker.value,
                self.minimum_perturb_factor,
                self.maximum_perturb_factor,
            )
            actor_fresh = deepcopy(self.actor)
            actor_fresh.apply(utils.weight_init)
            utils.soft_update_params(self.actor, actor_fresh, alpha, False)
            self.actor_opt.state.clear()
            self.reset_step = step

        return metrics

    def update_value(
        self, low_dim_obs, fused_view_feats, action, time_obs, updating_intrinsic_critic
    ):
        metrics = dict()
        critic = self.intr_critic if updating_intrinsic_critic else self.critic
        value = self.intr_value if updating_intrinsic_critic else self.value
        value_opt = self.intr_value_opt if updating_intrinsic_critic else self.value_opt
        qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
        if self.distributional_critic:
            qs = critic.from_dist(qs)
        q = qs.min(-1, keepdim=True)[0]
        v = value(low_dim_obs, fused_view_feats, time_obs)
        vf_err = v - q
        vf_weight = torch.where(
            vf_err > 0, (1 - self.exploitation_expectile), self.exploitation_expectile
        )
        vf_loss = (vf_weight * (vf_err**2)).mean()

        if self.logging:
            metrics["v_net"] = v.mean().item()
            metrics["vf_loss"] = vf_loss.item()

        value_opt.zero_grad(set_to_none=True)
        vf_loss.backward()
        value_opt.step()

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
        value = self.intr_value if updating_intrinsic_critic else self.value
        with torch.no_grad():
            std = self.get_std(step)
            dist = self.actor(next_low_dim_obs, next_fused_view_feats, std)
            next_action = dist.sample(clip=self.stddev_clip)
            target_qs = critic_target(
                next_low_dim_obs, next_fused_view_feats, next_action, next_time_obs
            )
            if self.distributional_critic:
                target_qs = critic_target.from_dist(target_qs)
            explore_q = target_qs.min(-1, keepdim=True)[0]
            # Dormant-ratio-guided exploitation
            exploit_v = value(next_low_dim_obs, next_fused_view_feats, next_time_obs)
            # Exploitation hyperparameter
            lam = self.exploitation_lam_max * self._drm_stddev(
                self.dormant_ratio_tracker.value,
                self.dormant_ratio_threshold,
                self.exploitation_temperature,
                False,
            )
            next_q = lam * exploit_v + (1 - lam) * explore_q
            target_q = reward + bootstrap * discount * next_q
            if self.distributional_critic:
                return critic_target.to_dist(target_q)
            return target_q

    # DrM update awaken step
    def update_awaken_step(self, step):
        if (
            self.training
            and self.awaken_step is None
            and self.dormant_ratio_tracker.value < self.dormant_ratio_threshold
        ):
            self.awaken_step = step

    @staticmethod
    def _drm_stddev(
        dormant_ratio, dormant_ratio_threshold, temperature, invert_dormant_delta
    ):
        dormant_delta = dormant_ratio - dormant_ratio_threshold
        if invert_dormant_delta:
            dormant_delta *= -1
        return 1 / (1 + np.exp(dormant_delta / temperature))

    def reset(self, step: int, agents_to_reset: list[int]):
        super().reset(step, agents_to_reset)
        for aid in agents_to_reset:
            self.value.reset(aid)
            if self.intr_value is not None:
                self.intr_value.reset(aid)

    def set_eval_env_running(self, value: bool):
        super().set_eval_env_running(value)
        self.value.set_eval_env_running(value)
        if self.intr_value is not None:
            self.intr_value.set_eval_env_running(value)

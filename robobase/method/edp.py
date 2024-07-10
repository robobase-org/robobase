from functools import partial
from typing_extensions import override
from copy import deepcopy
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from diffusers import (
    DDPMScheduler,
    SchedulerMixin,
    DDIMScheduler,
)
from diffusers.training_utils import EMAModel

from robobase.method.actor_critic import ActorCritic
from robobase.models.fully_connected import FullyConnectedModule
from robobase.method.actor_critic import Critic

# TODO
# Following the original implementaions, this class is only tested with D4RL
# environments.
# As D4RL environments are relatively simple with state-based observations, the
# following features are not tested and hence not supported for now.
# 1. distributional critic special case
# 2. intrinsic critic special case
# 3. priority replay special case
# 4. fused view feats and timeobs special case


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        temperature: float = 1.0,
        learnable_std: bool = False,
    ):
        super().__init__()
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self._temperature = temperature
        self._learnable_std = learnable_std
        if learnable_std:
            self._log_std = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, mean):
        if self._learnable_std:
            log_stds = torch.clamp(self._log_std, self._log_std_min, self._log_std_max)
            std = torch.exp(log_stds * self._temperature)
        else:
            std = torch.ones_like(mean, device=mean.device)

        return torch.distributions.normal.Normal(mean, std)


class IQLHelper:
    """
    A helper class to calculate IQL objectives.
    """

    def __init__(self, expectile: float, awr_temperature: float):
        self._awr_temperature = awr_temperature
        self._expectile = expectile

    def awr_loss(self, min_q, v, log_prob):
        # - Compute awr loss
        adv = torch.exp((min_q - v) * self._awr_temperature).clip(max=100.0)
        awr_loss = -(adv * log_prob).mean()

        return awr_loss, adv

    def critic_loss(self, qs, target_qs, v, min_q, loss_coeff):
        # - Compute IQL critic loss using expectile regression
        q_critic_loss = F.mse_loss(qs, target_qs.detach(), reduction="none").mean(
            -1, keepdim=True
        )
        vf_err = min_q.detach() - v
        vf_weight = torch.where(vf_err > 0, self._expectile, (1 - self._expectile))
        vf_loss = (vf_weight * (vf_err**2)).mean()
        critic_loss = (q_critic_loss + vf_loss) * loss_coeff.unsqueeze(1)
        critic_loss = torch.mean(critic_loss)

        return critic_loss, q_critic_loss, vf_loss


class Actor(nn.Module):
    def __init__(
        self,
        action_space: spaces.Box,
        actor_model: FullyConnectedModule,
        noise_scheduler: SchedulerMixin,
        num_diffusion_iters: int,
    ):
        super().__init__()
        assert len(action_space.shape) == 2
        self.action_space = action_space
        self.actor_model = actor_model
        self.noise_scheduler = noise_scheduler
        self.num_diffusion_iters = num_diffusion_iters
        self.sequence_length = action_space.shape[0]
        assert self.sequence_length == 1
        self.action_dim = action_space.shape[1]
        self.ema = EMAModel(
            model=self.actor_model,
            power=0.75,
        )

    @property
    def preferred_optimiser(self) -> callable:
        return getattr(
            self.actor_model,
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

    def add_noise_and_predict(self, obs, action) -> tuple[torch.Tensor, torch.Tensor]:
        # In the forward process, we add noise to action and predict the noise added
        # from the actor_model

        obs_features = obs["low_dim_obs"]

        b = obs_features.shape[0]  # batch size
        # sample noise to add to actions
        noise = torch.randn((b,) + self.action_space.shape, device=obs_features.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b,),
            device=obs_features.device,
        ).long()

        # add noise to the clean actions
        # according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        # Predict the noise
        net_ins = {
            "actions": noisy_actions,
            "features": obs_features,
            "timestep": timesteps,
        }
        noise_pred = self.actor_model(net_ins).view(noisy_actions.shape)
        return noise_pred, noise, noisy_actions, timesteps

    @torch.no_grad()
    def infer(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse process for inference. Denoise from noisy action.
        """
        # get obs features
        obs_features = self._combine(obs["low_dim_obs"], obs["fused_view_feats"])

        # ema averaged model
        actor_model = self.ema.averaged_model

        # initialize action from Gaussian noise
        b = obs_features.shape[0]
        noisy_action = torch.randn(
            (b, self.sequence_length, self.action_dim), device=obs_features.device
        )

        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        # denoise
        for k in self.noise_scheduler.timesteps:
            net_ins = {
                "actions": noisy_action,
                "features": obs_features,
                "timestep": k,
            }
            noise_pred = actor_model(net_ins)

            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action,
            ).prev_sample

        assert (noisy_action <= 1.001).all() and (noisy_action >= -1.001).all()
        return noisy_action

    def reset(self, env_index: int):
        self.actor_model.reset(env_index)

    def set_eval_env_running(self, value: bool):
        self.actor_model.eval_env_running = value


class DiffusionRL(ActorCritic):
    """
    This classes implements using diffusion as policy class for offline and online RL
    settings.

    The main challenge is to incorporate diffusion training loss with RL objectives.
    This class currently supports the following methods:
        1. Diffusion-QL (https://arxiv.org/abs/2208.06193) where the gradient of Q
        function is back-propogated through the diffusion chain to update model's
        parameter.
        2. Efficient Diffusion Policy from https://arxiv.org/abs/2305.20081 where the
        log_prob of actions sampled from current policy is approximated by constructing
        a Gaussian with mean equals to a (approximately) sampled action from the
        diffusion model. Then policy improvements is achieved via IQL with AWR.
        (https://arxiv.org/abs/2110.06169)
    """

    def __init__(
        self,
        num_diffusion_iters: int,
        solver_type: str = "DDPM",
        beta_schedule: str = "linear",
        diff_coeff=1.0,
        guide_coeff=1.0,
        actor_update_method: str = "iql",
        dql_alpha=2.0,
        expectile=0.5,
        awr_temperature=3.0,
        learnable_std: bool = False,
        *args,
        **kwargs,
    ):
        # sanity check
        if not kwargs["frame_stack_on_channel"]:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for diffusion policies."
            )

        # diffusion
        self.num_diffusion_iters = num_diffusion_iters
        if solver_type == "DDPM":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                # the choice of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule=beta_schedule,
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type="epsilon",
            )
        elif solver_type == "DDIM":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_diffusion_iters,
                # the choice of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule=beta_schedule,
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type="epsilon",
            )
        else:
            raise NotImplementedError(f"{solver_type} is not supported!!")

        super().__init__(*args, **kwargs)

        # TODO: Currently EDP is only tested with D4RL
        if self.intrinsic_reward_module:
            raise NotImplementedError("Intrinsic reward is not supported for EDP")
        if self.distributional_critic:
            raise NotImplementedError("Distributional critic is not supported for EDP")
        if self.use_pixels:
            raise NotImplementedError("Pixel observations is not supported for EDP")

        # Implicit Q-Learning loss function heler
        self._iql = IQLHelper(expectile, awr_temperature)

        # Create an additional value function used by IQL.
        # NOTE: Overwrite the critic_opt defined in super class
        self.value_fn, self.critic_opt = self._build_value(self.critic)
        self.intr_value_fn = self.intr_critic_opt = None

        # EDP
        # - prepare additional values for approximate action sampling
        # - 1 / sqrt(alpha_k)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(
            1.0 / self.noise_scheduler.alphas_cumprod
        ).to(self.device)
        # - sqrt(1 - alpha_k) / sqrt(alpha_k)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.noise_scheduler.alphas_cumprod - 1
        ).to(self.device)
        self.gaussion_policy = GaussianPolicy(learnable_std)
        self.gaussion_policy.to(self.device)

        # Extras
        self.diff_coeff = diff_coeff
        self.guide_coeff = guide_coeff
        self.actor_update_method = actor_update_method
        self.dql_alpha = dql_alpha

    def _build_value(self, critic):
        value_fn = Critic(deepcopy(self.critic.qs[0]), 1).to(self.device)
        critic_opt = torch.optim.Adam(
            list(critic.parameters()) + list(value_fn.parameters()),
            lr=self.critic_lr,
        )

        return value_fn, critic_opt

    @override
    def build_actor(self):
        self.actor_model = self.actor_model(
            input_shapes=self.get_fully_connected_inputs(),
            output_shape=self.action_space.shape[-1],
            num_envs=self.num_train_envs + self.num_eval_envs,
        ).to(self.device)
        self.actor = Actor(
            self.action_space,
            self.actor_model,
            self.noise_scheduler,
            self.num_diffusion_iters,
        ).to(self.device)
        self.actor_opt = (self.actor.preferred_optimiser)(lr=self.actor_lr)

    def get_fully_connected_inputs(self) -> dict[str, tuple]:
        # Get observation input shapes
        input_shapes = super().get_fully_connected_inputs()
        # Get action input shapes
        input_shapes["actions"] = self.action_space.shape[-1:]
        return input_shapes

    @override
    def _act(self, obs: dict[str, torch.Tensor], eval_mode: bool):
        low_dim_obs = fused_rgb_feats = None
        if self.low_dim_size > 0:
            low_dim_obs = self._act_extract_low_dim_state(obs)
        if self.use_pixels:
            rgb_obs = self._act_extract_rgb_obs(obs)
            with torch.no_grad():
                multi_view_rgb_feats = self.encoder(rgb_obs.float())
                fused_rgb_feats = self.view_fusion(multi_view_rgb_feats)
                if not self.frame_stack_on_channel:
                    fused_rgb_feats = fused_rgb_feats.view(
                        -1, self.time_dim, *fused_rgb_feats.shape[1:]
                    )

        # get denoised action
        obs = {"low_dim_obs": low_dim_obs, "fused_view_feats": fused_rgb_feats}
        if not eval_mode:
            denoised_action = self.actor.infer(obs)
        else:
            # Energy-based action selection during eval
            denoised_action = self.energy_based_action_selection(obs)

        return denoised_action

    def energy_based_action_selection(
        self, obs: Dict, num_samples: int = 50
    ) -> torch.Tensor:
        """
        Sample an action from learned policy, weighted by Q function.

        Args:
            obs (Dict): the observation
            num_samples (int): the number of actions to sample

        Return:
            (Tensor): the sampled action.
        """
        # Sample a batch of actions
        new_obs = {}
        new_obs["low_dim_obs"] = obs["low_dim_obs"].repeat(num_samples, 1)
        new_obs["fused_view_feats"] = None  # TODO: we do not support fused_view_feats
        denoised_action = self.actor.infer(new_obs)

        # Get q value of the sampled action
        qs = self.critic(
            new_obs["low_dim_obs"], new_obs["fused_view_feats"], denoised_action, None
        )
        min_q = qs.min(-1, keepdim=True)[0]

        # Sample action according to q values. Higher q value action will more likely
        # be sampled.
        idx = torch.distributions.categorical.Categorical(
            logits=min_q.view(-1)
        ).sample()
        selected_action = denoised_action[idx].unsqueeze(0)
        return selected_action

    @override
    def update_actor(
        self, low_dim_obs, fused_view_feats, action, step, time_obs, demos, loss_coeff
    ):
        # sanity check
        assert fused_view_feats is None and time_obs is None

        metrics = dict()

        obs = {
            "low_dim_obs": low_dim_obs,
            "fused_view_feats": fused_view_feats,
            "time_obs": time_obs,
        }

        # calculate diffusion loss
        # NOTE: loss_coeff might not be uniform here
        noise_preds, noises, _, _ = self.actor.add_noise_and_predict(obs, action)
        mse_loss = (
            F.mse_loss(noise_preds, noises, reduction="none")
            .mean(-1)
            .mean(-1, keepdims=True)
        )
        diff_loss = (mse_loss * loss_coeff.unsqueeze(1)).mean()

        # calculate guidance loss from RL objectives
        guide_loss = self.calc_edp_guide_loss(obs, action, metrics)

        # calculate total loss
        total_loss = self.diff_coeff * diff_loss + self.guide_coeff * guide_loss

        # Gradient descent
        self.actor_opt.zero_grad()
        total_loss.backward()
        if self.actor_grad_clip:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_clip
            )
        self.actor_opt.step()

        # update Exponential Moving Average of the model weights
        self.actor.ema.step(self.actor.actor_model)

        metrics["actor_grad_norm"] = actor_grad_norm.item()
        metrics["diff_loss"] = diff_loss.item()
        metrics["guide_loss"] = guide_loss.item()
        metrics["actor_loss"] = total_loss.item()
        return metrics

    @override
    def update_critic(
        self,
        low_dim_obs,
        fused_view_feats,
        actions,
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
        # sanity check
        assert not updating_intrinsic_critic
        assert fused_view_feats is None and time_obs is None

        lp = "intrinsic_" if updating_intrinsic_critic else ""

        # get relevant objects.
        critic, critic_opt, value_fn = self._get_critics()

        # calculate target_qs and qs
        target_qs = self.calculate_target_q(
            step,
            actions,
            next_low_dim_obs,
            next_fused_view_feats,
            next_time_obs,
            reward,
            discount,
            bootstrap,
            updating_intrinsic_critic,
        )
        target_qs = target_qs.repeat(1, self.num_critics)
        qs = critic(low_dim_obs, fused_view_feats, actions, time_obs)

        # calculate predicted value of state for iql
        dummy_actions = torch.zeros_like(actions)  # [bs, act_seq, act_dim]
        v = value_fn(low_dim_obs, fused_view_feats, dummy_actions, time_obs)
        min_q = qs.min(-1, keepdim=True)[0]

        # invoke iql
        critic_loss, q_critic_loss, vf_loss = self._iql.critic_loss(
            qs, target_qs, v, min_q, loss_coeff
        )

        # double q learning
        metrics = dict()
        metrics[f"{lp}critic_target_q"] = target_qs.mean().item()
        for i in range(self.num_critics):
            metrics[f"{lp}critic_q{i + 1}"] = qs[..., i].mean().item()
        metrics[f"{lp}critic_loss"] = critic_loss.item()
        metrics["vf_loss"] = vf_loss.item()

        # optimize encoder and critic
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.critic_grad_clip:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                critic.parameters(), self.critic_grad_clip
            )
        critic_opt.step()
        metrics["critic_grad_norm"] = critic_grad_norm.item()

        return metrics

    def calculate_target_q(
        self,
        step,
        actions,
        next_low_dim_obs,
        next_fused_view_feats,
        next_time_obs,
        reward,
        discount,
        bootstrap,
        updating_intrinsic_critic,
    ):
        # Get value_fn
        value_fn = self.value_fn

        with torch.no_grad():
            dummy_actions = torch.zeros_like(actions)  # [bs, act_seq, act_dim]
            v = value_fn(
                next_low_dim_obs, next_fused_view_feats, dummy_actions, next_time_obs
            )

            target_q = reward + bootstrap * discount * v
            return target_q

    def calc_edp_guide_loss(self, obs, action, metrics):
        # This is to approximate sampling an action from current policy. Instead of
        # performing the reverse process through diffusion from random noise, it
        # corrupts the current action into noisy action then perform one-step
        # reconstruction of denoised action.
        approx_action = self.approximate_action(obs, action)

        if self.actor_update_method == "iql":
            # Implicit q-learning loss
            guide_loss = self.calc_guide_loss_iql(obs, action, approx_action, metrics)
        elif self.actor_update_method == "dql":
            # Diffusion q-learning loss
            guide_loss = self.calc_guide_loss_dql(obs, action, approx_action, metrics)

        return guide_loss

    def calc_guide_loss_iql(self, obs, action, approx_action, metrics):
        def calc_awr_loss(critic, value_fn):
            # Calculate advantage
            with torch.no_grad():
                qs = critic(low_dim_obs, fused_view_feats, action, time_obs)
                dummy_actions = torch.zeros_like(action)  # [bs, act_seq, act_dim]
                v = value_fn(low_dim_obs, fused_view_feats, dummy_actions, time_obs)
                min_q = qs.min(-1, keepdim=True)[0]

            # Calculate log(pi(a|s))
            # EDP uses a Gaussion policy with mean=approx_action to approximate pi.
            dist = self.gaussion_policy(approx_action)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)

            # calc awr loss
            awr_loss, adv = self._iql.awr_loss(min_q, v, log_prob)

            # logging
            metrics["min_q"] = min_q.mean().item()
            metrics["v"] = v.mean().item()
            metrics["adv"] = adv.mean().item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
            metrics["awr_loss"] = awr_loss.item()

            return awr_loss

        low_dim_obs = obs["low_dim_obs"]
        fused_view_feats = obs["fused_view_feats"]
        time_obs = obs["time_obs"]

        # AWR objective: E_{(s,a)~D} [Adv * log(pi(a|s))]
        awr_loss = calc_awr_loss(self.critic, self.value_fn)

        return awr_loss

    def calc_guide_loss_dql(self, obs, action, approx_action, metrics):
        low_dim_obs = obs["low_dim_obs"]
        fused_view_feats = obs["fused_view_feats"]
        time_obs = obs["time_obs"]

        with torch.no_grad():
            qs = self.critic(low_dim_obs, fused_view_feats, approx_action, time_obs)

            # Random choose a q function
            x = random.uniform(0, 1)
            q = qs[..., 0] if x > 0.5 else qs[..., 1]

            lmbda = self.dql_alpha / torch.abs(q).mean()

        guide_loss = -lmbda * q.mean()
        return guide_loss

    def approximate_action(self, obs, action):
        # This is to approximate sampling an action from current policy. Instead of
        # performing the reverse process through diffusion from random noise, it
        # corrupts the current action into noisy action then perform one-step
        # reconstruction of denoised action.

        # 1. add noise to action and get noise prediction
        noise_preds, _, noisy_actions, timesteps = self.actor.add_noise_and_predict(
            obs, action
        )

        # 2. perform approximate one-step denoise in batch
        coeff1 = self._expand_tensor(
            self.sqrt_recip_alphas_cumprod[timesteps], noisy_actions
        )
        coeff2 = self._expand_tensor(
            self.sqrt_recipm1_alphas_cumprod[timesteps], noise_preds
        )
        approx_denoised_action = coeff1 * noisy_actions - coeff2 * noise_preds
        approx_denoised_action = approx_denoised_action.clip(-1, 1)

        return approx_denoised_action

    def _get_critics(self):
        return self.critic, self.critic_opt, self.value_fn

    def _expand_tensor(self, t, target_t):
        while len(t.shape) < len(target_t.shape):
            t = t.unsqueeze(-1)
        return torch.broadcast_to(t, target_t.shape)

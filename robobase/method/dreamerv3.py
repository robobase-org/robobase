from abc import ABC
from copy import deepcopy
from typing import Iterator, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Distribution

from robobase import utils
from robobase.method.core import ModelBasedMethod
from robobase.models.fusion import FusionMultiCamFeature
from robobase.models.encoder import EncoderModule
from robobase.models.decoder import DecoderModule
from robobase.models.fully_connected import FullyConnectedModule
from robobase.models.model_based.dynamics_model import DynamicsModelModule
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    extract_from_batch,
    stack_tensor_dictionary,
    extract_many_from_batch,
)
from robobase.models.model_based.distributions import TruncatedNormalWithScaling
from robobase.models.model_based.utils import (
    RequiresGrad,
    lambda_return,
    ReturnEMA,
    DistWrapperModule,
    BatchTimeInputDictWrapperModule,
    batch_time_forward,
)
from robobase.models.model_based.distributions import symlog


class Critic(nn.Module):
    def __init__(self, critic_model: BatchTimeInputDictWrapperModule, num_critics: int):
        super().__init__()
        self.qs = nn.ModuleList(
            [
                DistWrapperModule(deepcopy(critic_model), "symlog_disc")
                for _ in range(num_critics)
            ]
        )
        for q in self.qs:
            q.initialize_output_layer(utils.uniform_weight_init(0.0))

    def forward(self, feats):
        qs = [q(feats) for q in self.qs]
        return torch.cat(qs, -1)

    def compute_loss(self, feats, targets):
        loss = 0.0
        for q in self.qs:
            loss += q.compute_loss(feats, targets)
        return loss


class Actor(nn.Module):
    def __init__(
        self,
        actor_model: BatchTimeInputDictWrapperModule,
        min_std=0.1,
        max_std=1.0,
    ):
        super().__init__()
        self.actor = actor_model
        self._min_std = min_std
        self._max_std = max_std

    def forward(self, feats) -> Distribution:
        out = self.actor(feats)
        mean, std = torch.split(out, out.shape[-1] // 2, dim=-1)
        mean, std = mean.flatten(-2), std.flatten(-2)  # remove action sequence
        mean = torch.tanh(mean)
        std = (self._max_std - self._min_std) * torch.sigmoid(std) + self._min_std
        dist = TruncatedNormalWithScaling(mean, std)
        dist = torchd.Independent(dist, 1)
        return dist


class DreamerV3(ModelBasedMethod, ABC):
    """DreamerV3 agent (see https://arxiv.org/abs/2301.04104)

    Note that we assume we only use this agent for *continuous* control, so that
    it's not possible to use this class for discrete control tasks (e.g., Atari)
    """

    def __init__(
        self,
        num_explore_steps: int,
        actor_lr: float,
        critic_lr: float,
        world_model_lr: float,
        weight_decay: float,
        critic_target_interval: int,
        critic_target_tau: float,
        discount: float,
        discount_lambda: float,
        always_bootstrap: bool,
        num_critics: int,
        actor_grad: str,
        use_return_normalization: bool,
        use_input_low_dim_symlog: bool,
        aent_scale: float,
        actor_grad_clip: Optional[float],
        critic_grad_clip: Optional[float],
        world_model_grad_clip: Optional[float],
        horizon: int,
        action_sequence_network_type: str,
        use_torch_compile: bool,
        use_amp: bool,
        pixel_loss_scale: float,
        low_dim_loss_scale: float,
        reward_loss_scale: float,
        discount_loss_scale: float,
        dyn_loss_scale: float,
        rep_loss_scale: float,
        kl_free: float,
        bc_loss_scale: float,
        actor_model: Optional[FullyConnectedModule],
        critic_model: Optional[FullyConnectedModule],
        pixel_encoder_model: Optional[EncoderModule],
        pixel_decoder_model: Optional[DecoderModule],
        low_dim_encoder_model: Optional[FullyConnectedModule],
        low_dim_decoder_model: Optional[FullyConnectedModule],
        dynamics_model: Optional[DynamicsModelModule],
        reward_predictor_model: Optional[FullyConnectedModule],
        discount_predictor_model: Optional[FullyConnectedModule],
        *args,
        **kwargs,
    ):
        """Initialize DreamerV3 agent.

        Args:
            num_explore_steps: Number of initial exploration steps (uniform action)
            actor_lr: Learning rate for the actor
            critic_lr: Learning rate for the critic
            world_model_lr: Learning rate for the world model
            weight_decay: Weight decay for optimizer
            critic_target_interval: Interval for updating target critic
            critic_target_tau: Fraction of update magintudes for target critic
            discount: Discount factor in MDP
            discount_lambda: Lambda value used for computing the target value with GAE
            always_bootstrap: Do always bootstrap, ignoring terminal signal
            num_critics: Number of critics to use
            actor_grad: Type of actor gradient to use for training the actor,
                i.e.) reinforce or dynamics
            use_return_normalization: If True, normalize return with 5/95% percentiles
                which is used for training the actor
            use_input_low_dim_symlog: Apply symlog to low-dim observations
            aent_scale: Scale of entropy regularization for exploration
            actor_grad_clip: Gradient clipping magnitude for the actor
            critic_grad_clip: Gradient clipping magnitude for the critic
            world_model_grad_clip: Gradient clipping magnitude for the world model
            horizon: Horizon for imagination into the future
            action_sequence_network_type: Use 'mlp' or 'rnn' for action sequence
            use_torch_compile: Use 'torch.compile'. Latest torch version is recommended
            use_amp: Use autocast for float16 training. Might lead to unstable training
            pixel_loss_scale: Loss scale for pixel reconstruction
            low_dim_loss_scale: Loss scale for low dim observations reconstruction
            reward_loss_scale: Loss scale for reward prediction
            discount_loss_scale: Loss scale for discount prediction
            dyn_loss_scale: Loss scale for dynamics loss
            rep_loss_scale: Loss scale for representation loss
            kl_free: Free bit for KL loss
            bc_loss_scale: Loss scale for BC loss
            actor_model: RoboBase model class for the actor
            critic_model: RoboBase model class for the critic
            pixel_encoder_model: RoboBase model class for the pixel encoder
            pixel_decoder_model: RoboBase model class for the pixel decoder
            low_dim_encoder_model: RoboBase model class for the low dim encoder
            low_dim_decoder_model: RoboBase model class for the low dim decoder
            dynamics_model: RoboBase model class for the dynamics model (e.g., RSSM)
            reward_predictor_model: RoboBase model class for the reward predictor
            discount_predictor_model: RoboBase model class for the discount predictor
        """
        super().__init__(*args, **kwargs)
        self.num_explore_steps = num_explore_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.world_model_lr = world_model_lr
        self.weight_decay = weight_decay
        self.critic_target_interval = critic_target_interval
        self.critic_target_tau = critic_target_tau
        self.discount = discount
        self.discount_lambda = discount_lambda
        self.always_bootstrap = always_bootstrap
        self.num_critics = num_critics
        self.actor_grad = actor_grad
        self.use_return_normalization = use_return_normalization
        self.use_input_low_dim_symlog = use_input_low_dim_symlog
        self.aent_scale = aent_scale
        self.actor_grad_clip = actor_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.world_model_grad_clip = world_model_grad_clip
        self.horizon = horizon
        self.action_sequence_network_type = action_sequence_network_type.lower()
        self.use_torch_compile = use_torch_compile
        self.use_amp = use_amp
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.pixel_encoder_model = pixel_encoder_model
        self.pixel_decoder_model = pixel_decoder_model
        self.low_dim_encoder_model = low_dim_encoder_model
        self.low_dim_decoder_model = low_dim_decoder_model
        self.dynamics_model = dynamics_model
        self.reward_predictor_model = reward_predictor_model
        self.discount_predictor_model = discount_predictor_model
        if self.action_sequence_network_type not in ["rnn", "mlp"]:
            raise ValueError(
                f"action_sequence_network_type: {action_sequence_network_type} "
                "not supported."
            )
        if self.use_torch_compile:
            torch.set_float32_matmul_precision(
                "high"
            )  # accelerate training when using torch.compile with tensorcore
            torch._dynamo.config.cache_size_limit = (
                64  # need this to make torch.compile work
            )
        self.rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        self.use_pixels = len(self.rgb_spaces) > 0
        self.use_multicam_fusion = len(self.rgb_spaces) > 1
        self.world_model_modules = []
        (
            self.actor,
            self.critic,
            self.pixel_encoder,
            self.pixel_decoder,
            self.low_dim_encoder,
            self.low_dim_decoder,
            self.dynamics,
            self.reward_predictor,
            self.discount_predictor,
            self.view_fusion,
        ) = (None,) * 10
        self.build_encoder()
        self.build_view_fusion()
        self.build_dynamics()
        self.build_decoder()
        self.build_actor()
        (
            self.critic,
            self.critic_target,
            self.critic_opt,
            self.critic_ema,
        ) = self.build_critic()
        self._critic_target_update_cnt = 0
        if self.intrinsic_reward_module:
            # TODO: Modify the intrinsic reward module not to fold time into channel
            self._intrinsic_critic_target_update_cnt = 0
            raise NotImplementedError("Intrinsic module not supported yet for MBRL")

        self._world_model_params = sum(
            [list(module.parameters()) for module in self.world_model_modules], []
        )
        self.world_model_opt = torch.optim.AdamW(
            self._world_model_params,
            world_model_lr,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        self.world_model_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.loss_scales = {
            "low_dim": low_dim_loss_scale,
            "pixel": pixel_loss_scale,
            "reward": reward_loss_scale,
            "discount": discount_loss_scale,
            "dyn": dyn_loss_scale,
            "rep": rep_loss_scale,
            "bc": bc_loss_scale,
        }
        self.kl_free = kl_free

        # Internal state for episode rollout and world model update
        self.is_firsts = np.array([True for _ in range(self.num_train_envs + 1)])
        self.rollout_train_state = None
        self.rollout_eval_state = None

    def build_actor(self):
        """Builds the actor"""
        actor_model = BatchTimeInputDictWrapperModule(
            self.actor_model(
                input_shapes={"feats": (self.feat_dim,)},
                output_shape=(self.action_space.shape[-1] * 2,),  # *2 for mean/std
            ),
            "feats",
        )
        self.actor = Actor(actor_model).to(self.device)
        if self.use_torch_compile:
            self.actor = torch.compile(self.actor)
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.actor_lr,
            eps=1e-5,
            weight_decay=self.weight_decay,
        )
        self.actor_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.actor.requires_grad_(requires_grad=False)

    def build_critic(self) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
        """Builds the critic, critic_target, and ReturnEMA"""
        critic_cls = Critic
        critic_model = BatchTimeInputDictWrapperModule(
            self.critic_model(
                input_shapes={"feats": (self.feat_dim,)},
                output_shape=(255,),
            ),
            "feats",
        )
        critic = critic_cls(critic_model, self.num_critics)
        critic.to(self.device)
        critic_target = deepcopy(critic)
        critic_target.load_state_dict(critic.state_dict())
        critic_target.to(self.device)
        critic_opt = torch.optim.AdamW(
            critic.parameters(),
            lr=self.critic_lr,
            eps=1e-5,
            weight_decay=self.weight_decay,
        )
        critic_target.eval()
        return_normalizer = ReturnEMA()
        return_normalizer.to(self.device)
        critic.requires_grad_(requires_grad=False)
        critic_target.requires_grad_(requires_grad=False)
        # NOTE: No scaler for critic, as it often destabilizes training
        if self.use_torch_compile:
            critic = torch.compile(critic)
            critic_target = torch.compile(critic_target)
        return critic, critic_target, critic_opt, return_normalizer

    def build_encoder(self):
        """Builds the pixel or low-dimensional encoder"""
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            # Build encoder for pixels
            rgb_shapes = [s.shape for s in rgb_spaces.values()]
            assert np.all(
                [sh == rgb_shapes[0] for sh in rgb_shapes]
            ), "Expected all RGB obs to be same shape."

            num_views = len(rgb_shapes)
            assert (
                not self.frame_stack_on_channel
            ), "frame_stack_on_channel not supported"
            obs_shape = rgb_shapes[0][1:]  # [C, H, W]
            self.pixel_encoder = self.pixel_encoder_model(
                input_shape=(num_views, *obs_shape)
            )
            self.pixel_encoder.to(self.device)
            if self.use_torch_compile:
                self.pixel_encoder = torch.compile(self.pixel_encoder)
            self.world_model_modules.append(self.pixel_encoder)
            self.pixel_encoder.requires_grad_(requires_grad=False)

        self.low_dim_latent_size = 0
        if self.low_dim_size > 0:
            # Build encoder for low_dim_obs
            self.low_dim_encoder = BatchTimeInputDictWrapperModule(
                self.low_dim_encoder_model(
                    input_shapes={"low_dim_obs": (self.low_dim_size,)},
                ),
                "low_dim_obs",
            )
            self.low_dim_encoder.to(self.device)
            if self.use_torch_compile:
                self.low_dim_encoder = torch.compile(self.low_dim_encoder)
            self.world_model_modules.append(self.low_dim_encoder)
            self.low_dim_latent_size = self.low_dim_encoder.output_shape[-1]
            self.low_dim_encoder.requires_grad_(requires_grad=False)

    def build_decoder(self):
        """Builds the pixel or low-dimensional decoder"""
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            # Build decoder for pixels
            rgb_shapes = [s.shape for s in rgb_spaces.values()]
            assert np.all(
                [sh == rgb_shapes[0] for sh in rgb_shapes]
            ), "Expected all RGB obs to be same shape."

            num_views = len(rgb_shapes)
            assert (
                not self.frame_stack_on_channel
            ), "frame_stack_on_channel not supported"
            obs_shape = rgb_shapes[0][1:]  # [C, H, W]
            self.pixel_decoder = self.pixel_decoder_model(
                input_shape=(self.feat_dim,),
                output_shape=(num_views, *obs_shape),
            )
            self.pixel_decoder.to(self.device)
            self.pixel_decoder.initialize_output_layer(utils.uniform_weight_init(1.0))
            if self.use_torch_compile:
                self.pixel_decoder = torch.compile(self.pixel_decoder)
            self.world_model_modules.append(self.pixel_decoder)
            self.pixel_decoder.requires_grad_(requires_grad=False)

        if self.low_dim_size > 0:
            # Build decoder for low_dim_obs
            self.low_dim_decoder = BatchTimeInputDictWrapperModule(
                self.low_dim_decoder_model(
                    input_shapes={"feats": (self.feat_dim,)},
                    output_shape=(self.low_dim_size,),
                ),
                "feats",
            )
            self.low_dim_decoder.to(self.device)
            self.low_dim_decoder.initialize_output_layer(utils.uniform_weight_init(1.0))
            if self.use_torch_compile:
                self.low_dim_decoder = torch.compile(self.low_dim_decoder)
            self.world_model_modules.append(self.low_dim_decoder)
            self.low_dim_decoder.requires_grad_(requires_grad=False)

    def build_view_fusion(self):
        """Builds the view fusion model that handles multi-view RGB observations

        Note that, in Dreamer, RSSM should be capable of fusing multi-view inputs.
        So we always use non-trainable 'flatten' module as a default option
        """
        self.rgb_latent_size = 0
        if not self.use_pixels:
            return
        if self.use_multicam_fusion:
            self.view_fusion = FusionMultiCamFeature(
                input_shape=self.pixel_encoder.output_shape,
                mode="flatten",
            )
            self.view_fusion.to(self.device)
            if self.use_torch_compile:
                self.view_fusion = torch.compile(self.view_fusion)
            self.view_fusion_opt = None
            self.rgb_latent_size = self.view_fusion.output_shape[-1]
        else:
            # Input can be [B, T, ..] or [B, ..]
            self.view_fusion = lambda x: x[..., 0, :]
            self.rgb_latent_size = self.pixel_encoder.output_shape[-1]

    def build_dynamics(self):
        """Builds the dynamics model (RSSM) along with reward/discount predictor"""
        self.dynamics = self.dynamics_model(
            input_shape=(self.rgb_latent_size + self.low_dim_latent_size,),
            action_dim=np.prod(self.action_space.shape),
        )
        self.dynamics.to(self.device)
        if self.use_torch_compile:
            self.dynamics = torch.compile(self.dynamics)
        self.feat_dim = self.dynamics.output_shape[-1]

        self.discount_predictor = DistWrapperModule(
            BatchTimeInputDictWrapperModule(
                self.discount_predictor_model(
                    input_shapes={"feats": (self.feat_dim,)},
                    output_shape=(1,),
                ),
                "feats",
            ),
            dist="binary",
        )
        self.discount_predictor.to(self.device)
        self.discount_predictor.initialize_output_layer(utils.uniform_weight_init(1.0))
        if self.use_torch_compile:
            self.discount_predictor = torch.compile(self.discount_predictor)

        self.reward_predictor = DistWrapperModule(
            BatchTimeInputDictWrapperModule(
                self.reward_predictor_model(
                    input_shapes={"feats": (self.feat_dim,)},
                    output_shape=(255,),
                ),
                "feats",
            ),
            dist="symlog_disc",
        )
        self.reward_predictor.to(self.device)
        self.reward_predictor.initialize_output_layer(utils.uniform_weight_init(0.0))
        if self.use_torch_compile:
            self.reward_predictor = torch.compile(self.reward_predictor)
        self.world_model_modules.append(self.dynamics)
        self.world_model_modules.append(self.discount_predictor)
        self.world_model_modules.append(self.reward_predictor)
        self.dynamics.requires_grad_(requires_grad=False)
        self.discount_predictor.requires_grad_(requires_grad=False)
        self.reward_predictor.requires_grad_(requires_grad=False)

    @property
    def low_dim_size(self) -> int:
        low_dim_state_spec = extract_from_spec(
            self.observation_space, "low_dim_state", missing_ok=True
        )
        low_dim_in_size = 0
        if low_dim_state_spec is not None:
            low_dim_in_size = low_dim_state_spec.shape[-1]
        return low_dim_in_size

    def act(
        self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool
    ) -> torch.Tensor:
        """Get action from current observations

        Note that this class decides which internal state (train/eval) to use
        based on eval_mode value. And then overrides the state with the output state
        from the world model.

        Args:
            observations: Dictionary containing input tensors
            step: Global step from workspace
            eval_mode: Whether the agent is in training/eval rollout
        Returns:
            action: Action to take at input observations
        """
        state = self.rollout_eval_state if eval_mode else self.rollout_train_state
        if step < self.num_explore_steps and not eval_mode:
            action = self.random_explore_action
        else:
            state = self.rollout_eval_state if eval_mode else self.rollout_train_state
            with torch.no_grad():
                action, state = self._act(observations, eval_mode, state)
            if eval_mode:
                self.rollout_eval_state = state
            else:
                self.rollout_train_state = state
        return action

    def _act_extract_rgb_obs(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract RGB observations from input observations dictionary

        Args:
            observations: Dictionary containing input tensors
        Returns:
            rgb_obs: RGB observation of shape [B, T, V, C, H, W]
        """
        rgb_obs = extract_many_from_spec(observations, r"rgb.*")
        rgb_obs = stack_tensor_dictionary(rgb_obs, 2)  # [B, T, V, C, H, W]
        rgb_obs = rgb_obs[:, -1:]  # Use last timestep for rollout
        return rgb_obs

    def _act_extract_low_dim_state(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract low dimensional observations from input observations dictionary

        If self.use_input_low_dim_symlog is True, we apply symlog transformation.
        This is used in DreamerV3 to stabilize training across different domains
        where the scale of low dim observations can be very different

        Args:
            observations: Dictionary containing input tensors
        Returns:
            low_dim_obs: low dimensional observation of shape [B, T, C]
        """
        low_dim_obs = extract_from_spec(observations, "low_dim_state")
        low_dim_obs = low_dim_obs[:, -1:]  # Use last timestep for rollout
        if self.use_input_low_dim_symlog:
            low_dim_obs = symlog(low_dim_obs)
        return low_dim_obs

    def _act_extract_is_first(self, eval_mode: bool) -> torch.Tensor:
        """Extract is_first tensor from the internal state of the agent

        Args:
            eval_mode: Whether the agent is in training/eval rollout

        Return:
            is_first: torch.Tensor that denotes whether each sample is the first sample
                in rollout. Used for re-initializing the RSSM state.
        """
        if eval_mode:
            is_first = torch.from_numpy(self.is_firsts[-1:]).clone().to(self.device)
            self.is_firsts[-1:] = False
        else:
            is_first = torch.from_numpy(self.is_firsts[:-1]).clone().to(self.device)
            self.is_firsts[:-1] = False
        return is_first.float()

    def _act(
        self,
        observations: dict[str, torch.Tensor],
        eval_mode: bool,
        state: dict[str, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """See DreamerV3.act"""
        low_dim_obs = rgb_obs = None
        if self.low_dim_size > 0:
            low_dim_obs = self._act_extract_low_dim_state(observations)
        if self.use_pixels:
            rgb_obs = self._act_extract_rgb_obs(observations)
        is_first = self._act_extract_is_first(eval_mode)
        with torch.no_grad():
            embed = self.encode(low_dim_obs, rgb_obs)[0][:, 0, :]
        sample = not eval_mode

        if state is None:
            latent = self.dynamics.initial(len(embed), embed.device)
            action = torch.zeros(
                (len(embed), np.prod(self.action_space.shape)),
                device=embed.device,
            )
            state = latent, action
        latent, action = state

        latent, _ = self.dynamics.obs_step(latent, action, embed, is_first, sample)
        feat = self.dynamics.get_feat(latent)
        policy = self.actor(feat)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample()
        state = (latent, action)
        action = action.view(action.shape[0], *self.action_space.shape)
        return action, state

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
    ]:
        """Extract required data from batch dictionary

        Args:
            replay_iter: Iterator from a replay buffer

        Returns:
            tuple: (
                metrics: Dictionary containing training statistics
                batch: Dictionary that contains all the data
                prev_action: torch.Tensor for previous actions
                reward: torch.Tensor for reward
                is_first: torch.Tensor for is_first
                terminal: torch.Tensor for terminal
                truncated: torch.Tensor for truncated
                bootstrap: torch.Tensor for bootstrap
                demos: torch.Tensor for demo indicators or None if there are no demos
            )
        """
        metrics = dict()
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        prev_action = batch["action"].flatten(-len(self.action_space.shape))
        reward = batch["reward"]
        is_first = batch["is_first"].to(reward.dtype)
        # set first t as first; i.e. reset hidden state
        is_first[:, 0] = 1
        terminal = batch["terminal"].to(reward.dtype)
        truncated = batch["truncated"].to(reward.dtype)
        # 1. If not terminal and not truncated, we bootstrap
        # 2. If not terminal and truncated, we bootstrap
        # 3. If terminal and not truncated, we don't bootstrap
        # 4. If terminal and truncated,(e.g., success in last timestep)
        #    we don't bootstrap as terminal has a priortiy over truncated
        # In summary, we do not bootstrap when terminal; otherwise we do bootstrap
        bootstrap = 1.0 - terminal
        if self.always_bootstrap:
            # Override bootstrap to be 1
            bootstrap = torch.ones_like(bootstrap)
        demos = extract_from_batch(batch, "demo", missing_ok=True)
        if self.logging:
            metrics["batch_reward"] = reward.mean().item()

        return (
            metrics,
            batch,
            prev_action,
            reward,
            is_first,
            terminal,
            truncated,
            bootstrap,
            demos,
        )

    def extract_low_dim_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract low dimensional observations from batch
        See _act_extract_low_dim_obs for more details"""
        low_dim_obs = extract_from_batch(batch, "low_dim_state")
        if self.use_input_low_dim_symlog:
            low_dim_obs = symlog(low_dim_obs)
        return low_dim_obs

    def extract_pixels(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract RGB observations from batch
        See _act_extract_rgb_obs for more details"""
        # dict of {"cam_name": (B, T, 3, H, W)}
        rgb_obs = extract_many_from_batch(batch, r"rgb(?!.*?tp1)")
        metrics = {}
        if self.logging:
            # Get first batch item and last timestep
            for k, v in rgb_obs.items():
                metrics[k] = v[0, -1]
        # -> (B, T, V, 3, H, W)
        rgb_obs = stack_tensor_dictionary(rgb_obs, 2)
        return rgb_obs, metrics

    def encode(
        self, low_dim_obs: torch.Tensor, rgb_obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode raw low dimensional / RGB observations into embeddings

        Args:
            low_dim_obs: torch.Tensor for low dimensional observations
            rgb_obs: torch.Tensor for RGB observations

        Returns:
            outs: Concatenated embeddings
            outs_dict: Dictionary containing each embedding
        """
        outs, outs_dict = [], dict()
        if low_dim_obs is not None:
            low_dim_latent = self.low_dim_encoder(low_dim_obs)
            outs.append(low_dim_latent)
            outs_dict["low_dim_latent"] = low_dim_latent
        if rgb_obs is not None:
            multi_view_feats = batch_time_forward(self.pixel_encoder, rgb_obs.float())
            fused_view_feats = self.view_fusion(multi_view_feats)
            outs.append(fused_view_feats)
            outs_dict["fused_view_feats"] = fused_view_feats
        return torch.cat(outs, -1), outs_dict

    def update_critic(
        self,
        seq: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Update critic parameters.

        Args:
            seq: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`
            target: target value from imagined future trajectories

        Returns:
            metrics: dictionary that contains training statistics
        """
        metrics = dict()
        with RequiresGrad(self.critic):
            critic_loss, mets = self.critic_loss(seq, target)
            metrics.update(**mets)

            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()

        # update critic target
        if self._critic_target_update_cnt % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
        self._critic_target_update_cnt += 1
        return metrics

    def update_actor(
        self,
        start: Dict[str, torch.Tensor],
        bootstrap: torch.Tensor,
        prev_action: torch.Tensor,
        demos: Optional[torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, np.ndarray]]:
        """Update actor parameters.

        We first imagine future latent states via `imagine`, and label rewards.
        Then we compute target values from future latents, and compute actor loss.

        Args:
            start: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`
            bootstrap: torch.Tensor for whether to bootstrap or not
            prev_action: [B, T, A]-shaped prev_action tensor
            demos: If not None, [B, T]-shaped demo tensor

        Returns:
            seq: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`
            target: [H, B*T, ...]-shaped target return tensor.
            metrics: dictionary that contains training statistics
        """
        metrics = dict()
        with RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Imagination that gives future latents
                seq = self.imagine(start, 1 - bootstrap, self.horizon)
                with torch.no_grad():
                    # Label rewards in future latents
                    reward = self.reward_predictor(seq["feat"])
                    seq["reward"] = reward.squeeze(-1)
                # Compute target return V
                target, mets = self.calculate_target_v(seq)
                metrics.update(**mets)
                actor_loss, mets = self.actor_loss(seq, target)
                metrics.update(**mets)
                if self.loss_scales["bc"] > 0.0:
                    bc_loss, mets = self.bc_loss(start, prev_action, demos)
                    metrics.update(**mets)
                    actor_loss = actor_loss + self.loss_scales["bc"] * bc_loss

            self.actor_opt.zero_grad(set_to_none=True)
            self.actor_scaler.scale(actor_loss).backward()
            self.actor_scaler.unscale_(self.actor_opt)
            if self.actor_grad_clip:
                norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.actor_grad_clip
                )
                if self.logging:
                    metrics["actor_loss_norm"] = norm.item()
            self.actor_scaler.step(self.actor_opt)
            self.actor_scaler.update()
        if self.logging:
            metrics["actor_loss"] = actor_loss.mean().item()
        return seq, target, metrics

    def calculate_world_model_loss(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        prev_action: torch.Tensor,
        reward: torch.Tensor,
        is_first: torch.Tensor,
        bootstrap: torch.Tensor,
        state: Optional[dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Calculate loss for world model update.

        See `update_world_model` for arguments

        Returns:
            losses: dictionary containing the loss for updating the world model
            outs: dictionary containing the outputs from RSSM
            kl_value: KL value recorded while training the world model
        """
        discount = bootstrap * self.discount
        losses = dict()
        embeds = self.encode(low_dim_obs, rgb_obs)[0]
        post, prior = self.dynamics.observe(embeds, prev_action, is_first, state)
        feats = self.dynamics.get_feat(post)

        if low_dim_obs is not None:
            low_dim_obs_pred = self.low_dim_decoder(feats)
            losses["low_dim"] = (
                F.mse_loss(low_dim_obs_pred, low_dim_obs, reduction="none")
                .sum(-1)
                .mean()
            )
        if rgb_obs is not None:
            # + 0.5 trick is from DreamerV3
            rgb_obs_pred = batch_time_forward(self.pixel_decoder, feats) + 0.5
            losses["pixel"] = (
                F.mse_loss(
                    rgb_obs_pred,
                    rgb_obs.float() / 255.0,
                    reduction="none",
                )
                .sum([-3, -2, -1])
                .mean()
            )
        reward_loss = self.reward_predictor.compute_loss(feats, reward)
        discount_loss = self.discount_predictor.compute_loss(feats, discount)
        losses["reward"] = reward_loss.mean()
        losses["discount"] = discount_loss.mean()

        rep_loss, dyn_loss, kl_value = self.dynamics.kl_loss(post, prior)
        rep_loss = torch.mean(torch.clip(rep_loss, min=self.kl_free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=self.kl_free))
        losses["rep"] = rep_loss
        losses["dyn"] = dyn_loss

        outs = dict(embeds=embeds, feats=feats, post=post, prior=prior)
        return losses, outs, kl_value

    def update_world_model(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        prev_action: torch.Tensor,
        reward: torch.Tensor,
        is_first: torch.Tensor,
        bootstrap: torch.Tensor,
        state: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
        """Update world model parameters.

        Args:
            rgb_obs: [B, T, V, C, H, W]-shaped RGB input tensor
            low_dim_obs: [B, T, D]-shaped low dimensional input tensor
            prev_action: [B, T, A]-shpaed prev_action tensor
            reward: [B, T]-shaped reward tensor
            is_first: [B, T]-shaped is_first tensor
            bootstrap: [B, T]-shaped bootstrap tensor
            state: dictionary that contains RSSM state

        Returns:
            outs: dictionary containing the outputs from RSSM
            last_state: dictionary for RSSM state from the last timestep in batch
            metrics: dictionary that contains training statistics
        """
        metrics = dict()
        with RequiresGrad(self.world_model_modules):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                losses, outs, kl_value = self.calculate_world_model_loss(
                    rgb_obs,
                    low_dim_obs,
                    prev_action,
                    reward,
                    is_first,
                    bootstrap,
                    state,
                )
                model_loss = sum(self.loss_scales.get(k) * v for k, v in losses.items())

            self.world_model_opt.zero_grad(set_to_none=True)
            self.world_model_scaler.scale(model_loss).backward()
            self.world_model_scaler.unscale_(self.world_model_opt)
            if self.world_model_grad_clip:
                norm = nn.utils.clip_grad_norm_(
                    self._world_model_params, self.world_model_grad_clip
                )
                if self.logging:
                    metrics["model_loss_norm"] = norm.item()
            self.world_model_scaler.step(self.world_model_opt)
            self.world_model_scaler.update()

        if self.logging:
            metrics.update(
                {f"{name}_loss": value.item() for name, value in losses.items()}
            )
            metrics["model_kl"] = kl_value.mean().item()
            metrics["prior_ent"] = (
                self.dynamics.get_dist(outs["prior"]).entropy().mean().item()
            )
            metrics["post_ent"] = (
                self.dynamics.get_dist(outs["post"]).entropy().mean().item()
            )

        last_state = {k: v[:, -1].detach() for k, v in outs["post"].items()}
        return outs, last_state, metrics

    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        """Update DreamerV3 agent

        Update world model, actor, and critic. Then update EMA target critic

        Args:
            replay_iter: Iterator from a replay buffer
            step: Global step from workspace
            replay_buffer: (Optional) replay buffer from workspace

        Returns:
            metrics: Dictionary containing training statistics
        """
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            raise NotImplementedError("MBRL does not support PER")
        (
            metrics,
            batch,
            prev_action,
            reward,
            is_first,
            terminal,
            truncated,
            bootstrap,
            demos,
        ) = self.extract_batch(replay_iter)
        low_dim_obs = rgb_obs = None
        if self.low_dim_size > 0:
            low_dim_obs = self.extract_low_dim_state(batch)
        if self.use_pixels:
            rgb_obs, ext_matrics = self.extract_pixels(batch)
            metrics.update(ext_matrics)

        # Note that Dreamer does not use state for updating world model and is_first
        # at first timestep in batch are all set to one to initialize new states
        outs, _, mets_wm = self.update_world_model(
            rgb_obs, low_dim_obs, prev_action, reward, is_first, bootstrap, state=None
        )
        metrics.update(**mets_wm)

        seq, target, mets_actor = self.update_actor(
            outs["post"], bootstrap, prev_action, demos
        )
        metrics.update(**mets_actor)

        mets_critic = self.update_critic(seq, target)
        metrics.update(**mets_critic)

        return metrics

    def calculate_target_v(
        self,
        seq: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Computes lambda-return from the values predicted with critic functions.

        See High-Dimensional Continuous Control Using Generalized Advantage Estimation
        (https://arxiv.org/abs/1506.02438) for more details.

        Layout:
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2

        Args:
            seq: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`

        Returns:
            target: target value from imagined future trajectories
            metrics: dictionary containing training statistics
        """
        critic_target = self.critic_target

        metrics = {}
        reward = seq["reward"]
        disc = seq["discount"]

        values = critic_target(seq["feat"])
        value = values.min(-1, keepdim=False)[0]

        # Compute Lambda return
        target = lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.discount_lambda,
            axis=0,
        )
        if self.logging:
            metrics["critic_target"] = value.mean().item()
            metrics["critic_lambda_target"] = target.mean().item()
        return target, metrics

    def actor_loss(
        self,
        seq: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Computes actor loss with reinforce or dynamics objectives, and also
        adds entropy regularization for actor to encourage exploration.

        Here are brief explanations on each actor_grad algorithms:
        - reinforce: weights action log-likelihoods with advantages at each state
        - dynamics: backpropagates gradients through RSSM future imagination to
            update the actor to provide the actions that maximize returns

        Layout:
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2

        Args:
            seq: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`
            target: [H, B*T, ...]-shaped target return tensor.

        Returns:
            actor_loss: loss for updating the actor
            metrics: dictionary containing training statistics
        """
        critic_target, critic_ema = self.critic_target, self.critic_ema

        metrics = {}
        ret = target[1:]
        policy = self.actor(seq["feat"][:-2].detach())
        if self.actor_grad == "reinforce":
            base = critic_target(seq["feat"][:-2].detach()).min(-1, keepdim=True)[0]
            raw_advantage, raw_ret = ret - base, ret
            if self.use_return_normalization:
                offset, scale = critic_ema(ret)
                normed_ret = (ret - offset) / scale
                normed_base = (base - offset) / scale
                normed_advantage = normed_ret - normed_base
                advantage, ret = normed_advantage, normed_ret
            else:
                advantage, ret = raw_advantage, raw_ret
            action = seq["action"][1:-1].detach()
            objective = policy.log_prob(action) * advantage.detach()
        elif self.actor_grad == "dynamics":
            raw_ret = ret
            if self.use_return_normalization:
                offset, scale = critic_ema(ret)
                normed_ret = (ret - offset) / scale
                ret = normed_ret
            else:
                ret = raw_ret
            objective = ret
        else:
            raise NotImplementedError(self.actor_grad)

        # Entropy regularization
        ent = policy.entropy()
        objective += self.aent_scale * ent
        weight = seq["weight"].detach()
        actor_loss = -(weight[:-2] * objective).mean()

        if self.logging:
            metrics["return"] = ret.mean().item()
            metrics["actor_ent"] = ent.mean().item()
            metrics["policy_loss"] = actor_loss.mean().item()
            if self.actor_grad != "dynamics":
                metrics["advantage"] = advantage.mean().item()
                metrics["advantage_max"] = advantage.max(-1).values.mean().item()
            if self.use_return_normalization:
                metrics["raw_return"] = raw_ret.mean().item()
                if self.actor_grad != "dynamics":
                    metrics["raw_advantage"] = raw_advantage.mean().item()
                    metrics["raw_advantage_max"] = (
                        raw_advantage.max(-1).values.mean().item()
                    )
        return actor_loss, metrics

    def critic_loss(
        self, seq: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Computes critic loss that regresses target returns.

        Layout:
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2

        Args:
            seq: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`
            target: [H, B*T, ...]-shaped target return tensor.

        Returns:
            critic_loss: loss for updating the critic
            metrics: dictionary containing training statistics
        """
        metrics = dict()
        target = target.detach()
        weight = seq["weight"].detach()

        critic_pred = self.critic(seq["feat"][:-1].detach())
        critic_loss = self.critic.compute_loss(seq["feat"][:-1].detach(), target)
        critic_loss = (critic_loss * weight[:-1]).mean()
        if self.logging:
            metrics["critic"] = critic_pred.mean().item()
            metrics["critic_loss"] = critic_loss.item()
        return critic_loss, metrics

    def imagine(
        self,
        start: Dict[str, torch.Tensor],
        is_terminal: torch.Tensor,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Imagines future latents with the current policy from start state.
        NOTE: It's important to make gradient flow through actions.

        Args:
            start: A dictionary that contains RSSM state.
                e.g.) {"deter": [B, T, ..], ...}
            is_terminal: [B,T]-shaped is_first tensor
            horizon: Horizon for future imagination

        Returns:
            seq: Dictionary that contains imagined future trajectories
                e.g.) {"deter": [H, B*T, ..], ...} where H is the horizon
                of the future imagination
        """

        def flatten(x: torch.Tensor):
            return x.reshape([-1] + list(x.shape[2:]))

        # Flattens to [B*T, ...] shape as we imagine futures from every state
        start = {k: flatten(v).detach() for k, v in start.items()}
        start["feat"] = self.dynamics.get_feat(start)
        start["action"] = torch.zeros_like(self.actor(start["feat"]).mode())
        # Make a dictionary of lists to save future imaginations in the list
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            # Sample action for each timestep
            action = self.actor(seq["feat"][-1].detach()).rsample()
            # One step RSSM forward without access to input observation
            state = self.dynamics.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.dynamics.get_feat(state)
            # Save every future latent inside the list
            for key, value in {
                **state,
                "action": action,
                "feat": feat,
            }.items():
                seq[key].append(value)
        # Stack the list for each key
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        with torch.no_grad():
            disc = self.discount_predictor(seq["feat"].detach())
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).to(disc.dtype)
                true_first *= self.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
            seq["discount"] = disc.detach()
            # Shift discount factors because they imply whether the following state
            # will be valid, not whether the current state is valid.
            seq["weight"] = torch.cumprod(
                torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
            )
        return seq

    def bc_loss(
        self,
        start: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        demos: torch.Tensor,
    ) -> tuple[torch.tensor, dict[str, np.ndarray]]:
        """BC loss enforces actor outputs in demo states to be similar to demo actions

        Layout:
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3

        Args:
            seq: Dictionary containing [H, B*T, ...]-shaped torch tensors,
                which can be obtained from `imagine`
            demo: [B, T]-shaped demo tensor

        Returns:
            bc_loss: BC loss for updating the actor
            metrics: dictionary containing training statistics
        """
        if demos is not None and torch.sum(demos) > 0:
            # NOTE: We should shift one timestep to handle prev_action
            feats = self.dynamics.get_feat(start)
            bc_feat, bc_action = feats[:, :-1], prev_action[:, 1:]
            bc_dist = self.actor(bc_feat.contiguous().detach())
            bc_like = bc_dist.log_prob(bc_action)
            bc_loss = (-bc_like * demos[:, :-1]).sum() / demos[:, :-1].sum()
            metrics = {"bc_loss": bc_loss.mean().item()}
        else:
            bc_loss = torch.tensor(0.0, device=prev_action.device)
            metrics = {"bc_loss": 0.0}
        return bc_loss, metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        """Reset the internal RSSM state for DreamerV3 agent

        Here, we set is_first to True. Then `_act` method will re-initialize
        the internal state if is_first is set to True.
        """
        for aid in agents_to_reset:
            self.is_firsts[aid] = True

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value

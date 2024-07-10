import hydra
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from robobase import utils
from robobase.models.fusion import FusionMultiCamFeature
from robobase.models.encoder import (
    EncoderMultiViewVisionTransformer,
)
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.method.utils import (
    RandomShiftsAug,
    extract_many_from_spec,
)
from robobase.method.dreamerv3 import DreamerV3
from robobase.models.model_based.utils import (
    RequiresGrad,
    batch_time_forward,
    BatchTimeInputDictWrapperModule,
)


class MaskedWorldModel(DreamerV3):
    """(Multi-View) MaskedWorldModel agent (see https://arxiv.org/abs/2302.02408)"""

    def __init__(
        self,
        num_update_steps: int,
        mae_lr: float,
        mae_batch_size: Optional[int],
        mae_warmup: int,
        mae_grad_clip: Optional[float],
        mask_ratio: float,
        mae_use_augmentation: bool,
        num_mask_views: int,
        low_dim_mae_loss_scale: float,
        use_frozen_mae: bool,
        mae_pretrain_steps: int,
        world_model_pretrain_steps: int,
        actor_critic_pretrain_steps: int,
        mae_save_path: Optional[str],
        world_model_save_path: Optional[str],
        actor_critic_save_path: Optional[str],
        mae_load_path: Optional[str],
        world_model_load_path: Optional[str],
        actor_critic_load_path: Optional[str],
        *args,
        **kwargs,
    ):
        """Initialize MWM agent.

        Args:
            num_update_steps: Repeat update for N times (required for UTD > 1)
            mae_lr: Learning rate for the MAE
            mae_batch_size: If not None, subsample N images for MAE training
            mae_warmup: Warmup scheduling for MAE training
            mae_grad_clip: Gradient clipping magnitude for the MAE
            mask_ratio: Ratio of the masking inside each viewpoint
            mae_use_augmentation: Use pixel shift augmentation for training MAE
            num_mask_views: Number of viewpoints to randomly mask
            low_dim_mae_loss_scale: Loss scale for MAE training
            use_frozen_mae: Freeze MAE and skip MAE training (after pre-training)
            mae_pretrain_steps: If not 0, we pre-train MAE
                on initial seed frames (or demos)
            world_model_pretrain_steps: If not 0, we pre-train world model
                on initial seed frames (or demos)
            actor_critic_pretrain_steps: If not 0, we pre-train Actor Critic
                on initial seed frames (or demos)
            mae_save_path: Path to save pre-trained MAE, if needed.
                If not specified, we save the model on default hydra path.
            world_model_save_path: Path to save pre-trained world model, if needed.
                If not specified, we save the model on default hydra path.
            actor_critic_save_path: Path to save pre-trained actor critic, if needed.
                If not specified, we save the model on default hydra path.
            mae_load_path: Path to load pre-trained MAE, if needed.
                Note that setting this has a higher priority than pretrain_steps.
            world_model_load_path: Path to load pre-trained world model, if needed.
                Note that setting this has a higher priority than pretrain_steps.
            actor_critic_load_path: Path to load pre-trained actor critic, if needed.
                Note that setting this has a higher priority than pretrain_steps.
        """
        self.num_update_steps = num_update_steps
        self.mae_lr = mae_lr
        self.mae_batch_size = mae_batch_size
        self.mae_warmup = mae_warmup
        self.mae_grad_clip = mae_grad_clip
        self.mask_ratio = mask_ratio
        self.num_mask_views = num_mask_views
        self.use_frozen_mae = use_frozen_mae
        self.low_dim_mae_encoder = self.low_dim_mae_decoder = None
        super().__init__(*args, **kwargs)
        self.loss_scales["low_dim_mae"] = low_dim_mae_loss_scale
        self.mae_aug = RandomShiftsAug(pad=4) if mae_use_augmentation else lambda x: x

        self.mae_pretrain_steps = mae_pretrain_steps
        self.world_model_pretrain_steps = world_model_pretrain_steps
        self.actor_critic_pretrain_steps = actor_critic_pretrain_steps
        self._should_do_mae_pretrain = mae_pretrain_steps > 0
        self._should_do_world_model_pretrain = world_model_pretrain_steps > 0
        self._should_do_actor_critic_pretrain = actor_critic_pretrain_steps > 0

        self.should_do_pretrains = {
            "mae": self._should_do_mae_pretrain,
            "world_model": self._should_do_world_model_pretrain,
            "actor_critic": self._should_do_actor_critic_pretrain,
        }
        self.load_paths = {
            "mae": mae_load_path,
            "world_model": world_model_load_path,
            "actor_critic": actor_critic_load_path,
        }
        self.save_paths = {
            "mae": mae_save_path,
            "world_model": world_model_save_path,
            "actor_critic": actor_critic_save_path,
        }

    def build_encoder(self):
        """Builds MAE, MAE feature encoder, and optionally low-dimensional encoder

        1. We build Multi-View Masked Autoencoder. Note that MAE (pixel) encoder is not
        included world model optimization, and have a separate optimizer
        2. Then we build MAE feature encoder that takes MAE features as inputs
           e.g.) Image -> [MAE] -> MAE features -> [MAE feature encoder] -> RGB latent
        3. Build low-dim encoder if it's required(e.g. when using proprio states)
        """
        # Build MAE
        self._build_mae_encoder()

        # We build view fusion model here to allow for building a
        # low-dimensional encoder that takes view_fused_feats as inputs
        if self.use_multicam_fusion:
            self.view_fusion = FusionMultiCamFeature(
                input_shape=self.pixel_encoder.output_shape,
                mode="flatten",
            )
            self.view_fusion.to(self.device)
            if self.use_torch_compile:
                self.view_fusion = torch.compile(self.view_fusion)
            self.view_fusion_opt = None
            self.low_dim_mae_size = self.view_fusion.output_shape[-1]
        else:
            # Input can be [B, T, ..] or [B, ..]
            self.view_fusion = lambda x: x[..., 0, :]
            self.low_dim_mae_size = self.pixel_encoder.output_shape[-1]

        # Build low-dimensional MAE feature encoder
        self.low_dim_mae_encoder = BatchTimeInputDictWrapperModule(
            self.low_dim_encoder_model(
                input_shapes={"low_dim_mae_obs": (self.low_dim_mae_size,)},
            ),
            "low_dim_mae_obs",
        )
        self.low_dim_mae_encoder.to(self.device)
        if self.use_torch_compile:
            self.low_dim_mae_encoder = torch.compile(self.low_dim_mae_encoder)
        self.world_model_modules.append(self.low_dim_mae_encoder)
        self.rgb_latent_size = self.low_dim_mae_encoder.output_shape[-1]
        self.low_dim_mae_encoder.requires_grad_(requires_grad=False)

        # Build low-dimensional encoder
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
        """Builds MAE feature decoder, and optionally low-dimensional decoder"""
        # NOTE: There's no pixel encoder in MWM as MWM reconstructs MAE features
        # Build decoder for low-dimensional MAE feature
        self.low_dim_mae_decoder = BatchTimeInputDictWrapperModule(
            self.low_dim_decoder_model(
                input_shapes={"feats": (self.feat_dim,)},
                output_shape=(self.low_dim_mae_size,),
            ),
            "feats",
        )
        self.low_dim_mae_decoder.to(self.device)
        if self.use_torch_compile:
            self.low_dim_mae_decoder = torch.compile(self.low_dim_mae_decoder)
        self.world_model_modules.append(self.low_dim_mae_decoder)
        self.low_dim_mae_decoder.requires_grad_(requires_grad=False)

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
            if self.use_torch_compile:
                self.low_dim_decoder = torch.compile(self.low_dim_decoder)
            self.world_model_modules.append(self.low_dim_decoder)
            self.low_dim_decoder.requires_grad_(requires_grad=False)

    def build_view_fusion(self):
        """Overrides `build_view_fusion` in super class (DreamerV3)

        MWM handles this in build_encoder. See `build_encoder`
        """
        pass

    def _build_mae_encoder(self):
        """Build Multi-View MAE"""
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        assert len(rgb_spaces) > 0, "MWM is visuo-motor control algorithm"
        rgb_shapes = [s.shape for s in rgb_spaces.values()]
        assert np.all(
            [sh == rgb_shapes[0] for sh in rgb_shapes]
        ), "Expected all RGB obs to be same shape."
        num_views = len(rgb_shapes)
        assert not self.frame_stack_on_channel, "frame_stack_on_channel not supported"
        obs_shape = rgb_shapes[0][1:]  # [C, H, W]
        self.pixel_encoder = self.pixel_encoder_model(
            input_shape=(num_views, *obs_shape)
        )
        assert isinstance(
            self.pixel_encoder, EncoderMultiViewVisionTransformer
        ), self.pixel_encoder
        self.pixel_encoder.to(self.device)
        if self.use_torch_compile:
            self.pixel_encoder = torch.compile(self.pixel_encoder)
        self.pixel_encoder.requires_grad_(requires_grad=False)
        self.mae_num_patches = self.pixel_encoder.num_patches
        self.mae_opt = torch.optim.AdamW(
            self.pixel_encoder.parameters(),
            lr=self.mae_lr,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        self.mae_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.mae_opt,
            lambda step: np.clip(step / max(self.mae_warmup, 1), 0.0, 1.0),
        )
        self.mae_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode(
        self, low_dim_obs: torch.Tensor, rgb_obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode raw low dimensional / RGB observations into embeddings

        Note that we detach fused_view_feats to make sure that gradient from
        world model training does not flow to pixel encoder (i.e., MAE)

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
        with torch.no_grad():
            # Gradient does not flow to MAE (pixel) encoder
            multi_view_feats = batch_time_forward(self.pixel_encoder, rgb_obs.float())
            fused_view_feats = self.view_fusion(multi_view_feats)
            fused_view_feats = fused_view_feats.detach()
        # Encode MAE features with MLP encoder
        low_dim_mae_latent = self.low_dim_mae_encoder(fused_view_feats)
        outs.append(low_dim_mae_latent)
        outs_dict["fused_view_feats"] = fused_view_feats
        return torch.cat(outs, -1), outs_dict

    def update_mae(
        self,
        rgb_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Update MAE parameters

        Args:
            rgb_obs: [B, T, V, C, H, W]-shaped RGB input tensor
            reward: [B, T]-shaped reward tensor

        Returns:
            metrics dictionary that contains training statistics
        """
        metrics = dict()
        rgb_obs = rgb_obs.reshape((-1,) + rgb_obs.shape[2:])
        reward = reward.reshape(-1, 1)
        if self.mae_batch_size:
            # Do subsample for compute-efficiency
            indices = torch.randperm(rgb_obs.shape[0])[: self.mae_batch_size]
            rgb_obs = rgb_obs[indices]
            reward = reward[indices]
        # Do augmentation if required
        shape = rgb_obs.shape
        rgb_obs = self.mae_aug(rgb_obs.view(-1, *shape[2:]).float())
        rgb_obs = rgb_obs.view(*shape)
        with RequiresGrad(self.pixel_encoder):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                mae_loss, mae_mets = self.pixel_encoder.calculate_loss(
                    rgb_obs,
                    reward,
                    self.mask_ratio,
                    self.num_mask_views,
                    return_mets=True,
                )
                metrics.update(**mae_mets)
            self.mae_scheduler.step()
            self.mae_opt.zero_grad(set_to_none=True)
            self.mae_scaler.scale(mae_loss).backward()
            self.mae_scaler.unscale_(self.mae_opt)
            if self.mae_grad_clip:
                norm = nn.utils.clip_grad_norm_(
                    self.pixel_encoder.parameters(), self.mae_grad_clip
                )
                if self.logging:
                    metrics["mae_loss_norm"] = norm.item()
            self.mae_scaler.step(self.mae_opt)
            self.mae_scaler.update()
        return metrics

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

        Difference to DreamerV3 is that we have MAE feature reconstruction objective
        and don't have pixel reconsturction objective.
        See base class for more details.
        """
        discount = bootstrap * self.discount
        losses = dict()
        embeds, embeds_dict = self.encode(low_dim_obs, rgb_obs)
        post, prior = self.dynamics.observe(embeds, prev_action, is_first, state)
        feats = self.dynamics.get_feat(post)

        # MAE feature reconstruction objective
        low_dim_mae_pred = self.low_dim_mae_decoder(feats)
        losses["low_dim_mae"] = (
            F.mse_loss(
                low_dim_mae_pred, embeds_dict["fused_view_feats"], reduction="none"
            )
            .sum(-1)
            .mean()
            .div_(self.mae_num_patches)
        )

        if low_dim_obs is not None:
            # Low-dimensional state reconstruction objective
            low_dim_obs_pred = self.low_dim_decoder(feats)
            losses["low_dim"] = (
                F.mse_loss(low_dim_obs_pred, low_dim_obs, reduction="none")
                .sum(-1)
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

    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> dict[str, np.ndarray]:
        """Update MWM agent

        Update mae, world model, actor, and critic. Then update EMA target critic.
        We also try to pre-train MAE using initial seed frames, if required.

        Args:
            replay_iter: Iterator from a replay buffer
            step: Global step from workspace
            replay_buffer: (Optional) replay buffer from workspace

        Returns:
            metrics: Dictionary containing training statistics
        """
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            raise NotImplementedError("MBRL does not support PER")

        for _ in range(self.num_update_steps):  # high UTD support via for loop
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
            rgb_obs, ext_metrics = self.extract_pixels(batch)
            metrics.update(ext_metrics)

            mets_mae = self.update_mae(rgb_obs, reward)
            metrics.update(**mets_mae)

            # Note that Dreamer does not use state for updating world model and is_first
            # at first timestep in batch are all set to one to initialize new states
            outs, _, mets_wm = self.update_world_model(
                rgb_obs,
                low_dim_obs,
                prev_action,
                reward,
                is_first,
                bootstrap,
                state=None,
            )
            metrics.update(**mets_wm)

            seq, target, mets_actor = self.update_actor(
                outs["post"], bootstrap, prev_action, demos
            )
            metrics.update(**mets_actor)

            mets_critic = self.update_critic(seq, target)
            metrics.update(**mets_critic)

            # do pretraining if required
            self.try_pretrain(replay_iter)

        return metrics

    def try_pretrain(self, replay_iter: Iterator[dict[str, torch.Tensor]]):
        """Try to (i) load pre-trained parameters or (ii) do pre-training

        Args:
            replay_iter: Iterator from a replay buffer
        """
        for key in ["mae", "world_model", "actor_critic"]:
            if self.load_paths[key]:
                self.load_pretrained_model(key)
                self.should_do_pretrains[key] = False
                self.load_paths[key] = None
            if self.should_do_pretrains[key]:
                if key == "mae":
                    self.pretrain_mae(replay_iter)
                elif key == "world_model":
                    self.pretrain_world_model(replay_iter)
                elif key == "actor_critic":
                    self.pretrain_actor_critic(replay_iter)
                self.save_pretrained_model(key)
                self.should_do_pretrains[key] = False

    def pretrain_mae(self, replay_iter: Iterator[dict[str, torch.Tensor]]):
        """Pre-train MAE using initial seed frames

        Args:
            replay_iter: Iterator from a replay buffer
        """
        pbar = tqdm(range(self.mae_pretrain_steps), dynamic_ncols=True)
        print("Starting MAE Pre-training. Enter Q for early termination")
        for _ in range(self.mae_pretrain_steps):
            (_, batch, _, reward, *_) = self.extract_batch(replay_iter)
            rgb_obs, _ = self.extract_pixels(batch)
            mets_mae = self.update_mae(rgb_obs, reward)
            pbar.set_postfix(mets_mae)
            pbar.update(1)
            if utils.check_for_kill_input():
                break

    def pretrain_world_model(self, replay_iter: Iterator[dict[str, torch.Tensor]]):
        """Pre-train World Model using initial seed frames (or demos)

        Args:
            replay_iter: Iterator from a replay buffer
        """
        pbar = tqdm(range(self.world_model_pretrain_steps), dynamic_ncols=True)
        print("Starting World Model Pre-training. Enter Q for early termination")
        for _ in range(self.world_model_pretrain_steps):
            (
                _,
                batch,
                prev_action,
                reward,
                is_first,
                _,
                _,
                bootstrap,
                _,
            ) = self.extract_batch(replay_iter)
            low_dim_obs = rgb_obs = None
            if self.low_dim_size > 0:
                low_dim_obs = self.extract_low_dim_state(batch)
            rgb_obs, _ = self.extract_pixels(batch)
            _, _, mets_wm = self.update_world_model(
                rgb_obs,
                low_dim_obs,
                prev_action,
                reward,
                is_first,
                bootstrap,
                state=None,
            )
            pbar.set_postfix(mets_wm)
            pbar.update(1)
            if utils.check_for_kill_input():
                break

    def pretrain_actor_critic(self, replay_iter: Iterator[dict[str, torch.Tensor]]):
        """Pre-train Actor Critic using initial seed frames (or demos)

        Args:
            replay_iter: Iterator from a replay buffer
        """
        pbar = tqdm(range(self.world_model_pretrain_steps), dynamic_ncols=True)
        print("Starting Actor Critic Pre-training. Enter Q for early termination")
        for _ in range(self.world_model_pretrain_steps):
            (
                _,
                batch,
                prev_action,
                _,
                is_first,
                _,
                _,
                bootstrap,
                demos,
            ) = self.extract_batch(replay_iter)
            low_dim_obs = rgb_obs = None
            if self.low_dim_size > 0:
                low_dim_obs = self.extract_low_dim_state(batch)
            rgb_obs, _ = self.extract_pixels(batch)

            metrics = dict()
            with torch.no_grad():
                embeds = self.encode(low_dim_obs, rgb_obs)[0]
                post, _ = self.dynamics.observe(
                    embeds, prev_action, is_first, state=None
                )
            seq, target, mets_actor = self.update_actor(
                post, bootstrap, prev_action, demos
            )
            mets_critic = self.update_critic(seq, target)
            metrics.update(**mets_actor)
            metrics.update(**mets_critic)

            pbar.set_postfix(metrics)
            pbar.update(1)
            if utils.check_for_kill_input():
                break

    def save_pretrained_model(self, module: str):
        """Save the parameters of pre-trained models.

        If save_path is not specified, we save the model parameters in the default
        hydra workspace directory

        Args:
            module: "mae" or "world_model" or "actor_critic"
        """
        if module == "mae":
            models = [self.pixel_encoder]
            path = self.save_paths["mae"]
        elif module == "world_model":
            models = self.world_model_modules
            path = self.save_paths["world_model"]
        elif module == "actor_critic":
            models = [self.actor, self.critic, self.critic_target]
            path = self.save_paths["actor_critic"]
        else:
            raise ValueError(module)

        if path is None:
            path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        snapshot = Path(path) / f"{module}.pt"

        payload = [model.state_dict() for model in models]
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        logging.info(f"Saved pre-trained {module.upper()} in {str(snapshot)}")

    def load_pretrained_model(self, module: str):
        """Loads the pre-trained parameters for MAE, World Model, and Actor Critic

        Args:
            module: "mae" or "world_model" or "actor_critic"
        """
        if module == "mae":
            models = [self.pixel_encoder]
            path = self.load_paths["mae"]
        elif module == "world_model":
            models = self.world_model_modules
            path = self.load_paths["world_model"]
        elif module == "actor_critic":
            models = [self.actor, self.critic, self.critic_target]
            path = self.load_paths["actor_critic"]
        else:
            raise ValueError(module)
        snapshot = Path(path)
        with snapshot.open("rb") as f:
            state_dicts = torch.load(f, map_location="cpu")
            for model, state_dict in zip(models, state_dicts):
                if not self.use_torch_compile:
                    # Need to remove unwanted prefix when not using torch.compile
                    unwanted_prefix = "_orig_mod."
                    for k, _ in list(state_dict.items()):
                        if unwanted_prefix in k:
                            state_dict[k.replace("._orig_mod", "")] = state_dict.pop(k)
                    model.load_state_dict(state_dict)
            logging.info(f"Loaded pre-trained {module.upper()} in {str(snapshot)}")

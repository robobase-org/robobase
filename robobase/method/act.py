from typing_extensions import override

import torch
import torch.nn as nn
import numpy as np

try:
    from torchvision.transforms import v2 as tvf
except Exception:
    import torchvision.transforms as tvf  # Bamboo server use torchvision==0.14.1

from robobase.models import RoboBaseModule
from robobase.method.bc import BC
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.method.utils import (
    extract_from_spec,
    extract_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
    extract_many_from_batch,
)
from robobase.models.act.backbone import build_backbone, build_film_backbone


class ImageEncoderACT(RoboBaseModule):
    """
    Image Encoder for ACT model.

    Args:
        input_shape (Tuple[int, int, int, int]): Shape (views, channels, H, W)
        hidden_dim (int): Hidden dimension.
        position_embedding (str): Type of position embedding.
        lr_backbone (float): Learning rate for the backbone.
        masks (bool): Use masks.
        backbone (str): Backbone architecture.
        dilation (bool): Use dilation.
        use_lang_cond (bool): Use backbone with film for language conditioning

    Attributes:
        output_shape: The output shape of the encoder

    """

    # Mean and std for Resnet
    # Refer to https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
    VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
    VISUAL_OBS_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        hidden_dim: int = 512,
        position_embedding: str = "sine",
        lr_backbone: float = 1e-5,
        masks: bool = False,
        backbone: str = "resnet18",
        dilation: bool = False,
        use_lang_cond: bool = False,
    ):
        super().__init__()
        assert (
            len(input_shape) == 4
        ), f"Expected shape (V, C, H, W), but got {input_shape}"
        self._input_shape = tuple(input_shape)
        self._output_shape = None  # Will get calculated the first time output_shape
        # property gets called.

        self.use_lang_cond = use_lang_cond
        if self.use_lang_cond:
            self.backbone = build_film_backbone(
                hidden_dim=hidden_dim,
                position_embedding=position_embedding,
                backbone=backbone,
            )
        else:
            self.backbone = build_backbone(
                hidden_dim=hidden_dim,
                position_embedding=position_embedding,
                lr_backbone=lr_backbone,
                masks=masks,
                backbone=backbone,
                dilation=dilation,
            )

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        )

        if self.use_lang_cond:
            self.proj_text_emb = nn.Linear(self.backbone.num_channels, hidden_dim)

    @property
    def output_shape(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        """Return output shapes excluding the batch dimension."""
        if self._output_shape is None:
            bs = 1
            x = torch.randn((bs,) + self._input_shape).to(
                next(self.backbone.parameters()).device
            )
            task_emb = None
            if self.use_lang_cond:
                task_emb = torch.randn((bs, self.backbone.num_channels)).to(
                    next(self.backbone.parameters()).device
                )
            with torch.no_grad():
                feat, pos, task_emb = self.forward(x, task_emb=task_emb)
                if self.use_lang_cond:
                    self._output_shape = (
                        feat[0].shape,
                        pos[0].shape,
                        task_emb[0].shape,
                    )
                else:
                    self._output_shape = (feat[0].shape, pos[0].shape, None)

        return self._output_shape

    def forward(
        self, x: torch.Tensor, task_emb: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            self._input_shape == x.shape[1:]
        ), f"expected input shape {self._input_shape} but got {x.shape[1:]}"
        if self.use_lang_cond:
            task_emb = self.proj_text_emb(task_emb)

        all_cam_features = []
        all_cam_pos = []
        shape = x.shape
        for cam_id in range(self._input_shape[0]):
            # (b, v, fs, c, h, w) -> (b*fs, c, h, w)
            cur_x = x[:, cam_id].reshape(-1, 3, *self._input_shape[2:])

            # feat: (b*fs, c, h, w) -> (b*fs, feat_dim, 3, 3)
            if self.use_lang_cond:
                feat, pos = self.backbone(cur_x, task_emb=task_emb)
            else:
                feat, pos = self.backbone(cur_x)

            # feat: (b*fs, feat_dim, 3, 3) -> (b*fs, hidden_dim, 3, 3)
            feat = self.input_proj(feat[0])
            # pos: (b, pos_feat_dim, 3, 3)
            pos = pos[0]

            all_cam_features.append(feat)
            all_cam_pos.append(pos)

        # (b*fs, hidden_dim, 3, 3) -> (b*fs, hidden_dim, 3, 3*v)
        img_feat = torch.cat(all_cam_features, axis=3)

        # (b*fs, hidden_dim, 3, 3) -> (b, fs*hidden_dim, 3, 3*v)
        img_feat = img_feat.reshape(shape[0], -1, *img_feat.shape[2:])

        # (b, pos_feat_dim, 3, 3*v)
        pos = torch.cat(all_cam_pos, axis=3)

        return img_feat, pos, task_emb


class ACTPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        actor_model: RoboBaseModule = None,
        encoder_model: RoboBaseModule = None,
        view_fusion_model: RoboBaseModule = None,
    ):
        """
        Action Chuncking with Transformer (ACT) Policy.

        Args:
            observation_space: observation space
            actor_model (RoboBaseModule): The actor model.
            encoder_model (RoboBaseModule): The image encoder model.
            view_fusion_model (RoboBaseModule): The view fusion model.
        """
        super().__init__()

        self.actor_model = actor_model
        self.encoder_model = encoder_model
        self.view_fusion_model = view_fusion_model

        # Image normalization for ACT
        self.img_normalizer = tvf.Normalize(
            mean=ImageEncoderACT.VISUAL_OBS_MEAN, std=ImageEncoderACT.VISUAL_OBS_STD
        )

        # Used to merge frame stacks into single frame. Only used if frame_stack > 1.
        # NOTE: this is needed as original ACT does not support frame stacks
        # NOTE: self.encoder_model.output_shape contains
        #       (visual_obs_feat_shape, pos_emb_shape, task_emb_shape) where
        #       visual_obs_feat_shape = (fs*hidden_dim, 3, 3*v)
        # TODO: Try concatenating frame stacks in the last dimension, treating it
        #       as more views.
        key = list(observation_space.keys())[0]
        visual_obs_feat_dim = self.encoder_model.output_shape[0][0]
        self.frame_stack = observation_space[key].shape[0]
        target_feat_dim = int(visual_obs_feat_dim / self.frame_stack)
        self.projection_layer = nn.Sequential(
            nn.Conv2d(visual_obs_feat_dim, target_feat_dim, 1),
            nn.Mish(),
            nn.Conv2d(target_feat_dim, target_feat_dim, 1),
            nn.Mish(),
        )

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor = None,
        is_pad: torch.Tensor = None,
        task_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            qpos (torch.Tensor): joint positions.
            image (torch.Tensor): Input image data (unnormalized).
            actions (torch.Tensor, optional): Actions. Default is None.
            is_pad (torch.Tensor, optional): Whether actions are padded.
                Default is None.
            task_emb: task instruction embeddings. Default is None
        """
        # preprocess image
        # colour normalization
        # (b, v, fs*c, h, w) -> (b, v*fs, c, h, w)
        image = image.reshape(
            image.shape[0],
            image.shape[1] * self.frame_stack,
            3,
            image.shape[3],
            image.shape[4],
        )
        image /= 255.0
        image = self.img_normalizer(image)
        # (b, v*fs, c, h, w) -> (b, v, fs*c, h, w)
        image = image.reshape(
            image.shape[0], -1, 3 * self.frame_stack, image.shape[3], image.shape[4]
        )

        # feat: (b, fs * hidden_dim, 3, 3*v)
        feat, pos, task_emb = self.encoder_model(image, task_emb)

        # feat: (b, fs * hidden_dim, 3, 3*v) -> (b, hidden_dim, 3, 3*v)
        # NOTE: the detr_vae used by ACT expects views to be on the width channel.
        if self.frame_stack > 1:
            x = (
                self.projection_layer(feat),
                pos,
            )  # pass through projection layer to reduce the channel dimension
        else:
            x = (feat, pos)  # If frame_stack == 1, directly use the raw feature.

        # If actions is not None, we are training
        if actions is not None:
            x = self.actor_model(
                x, qpos, actions=actions, is_pad=is_pad, task_emb=task_emb
            )
            loss, loss_dict = self.actor_model.calculate_loss(
                x, actions=actions, is_pad=is_pad
            )
            return loss_dict

        # else we are at inference time
        else:
            x = self.actor_model(
                x, qpos, actions=actions, is_pad=is_pad, task_emb=task_emb
            )
            return x[0]


class ActBCAgent(BC):
    def __init__(
        self,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        use_lang_cond: bool = False,
        *args,
        **kwargs,
    ):
        """
        ACT Behavioral Cloning (BC) Agent.

        Args:
            lr (float): Learning rate for the policy.
            lr_backbone (float): Learning rate for the backbone.
            weight_decay (float): Weight decay for optimization.
        """
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        super().__init__(*args, **kwargs)

        # sanity check
        assert self.frame_stack_on_channel, "frame_stack_on_channel must be enabled"

        self.train()

    def build_actor(self):
        # NOTE: Encoder returns visual_obs_feat, pos_emb, task_emb, we pass
        # visual_obs_feat shape into actor model constructor
        self.actor_model = self.actor_model(
            input_shape=self.encoder.output_shape[0],
            state_dim=np.prod(self.observation_space["low_dim_state"].shape),
            action_dim=self.action_space.shape[-1],
        ).to(self.device)

        self.actor = ACTPolicy(
            self.observation_space,
            actor_model=self.actor_model,
            encoder_model=self.encoder,
        ).to(self.device)

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.actor.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.actor.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]

        self.actor_opt = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    @override
    def act(self, obs: dict[str, torch.Tensor], step: int, eval_mode: bool):
        if self.low_dim_size > 0:
            qpos = flatten_time_dim_into_channel_dim(
                extract_from_spec(obs, "low_dim_state")
            )
            qpos = qpos.detach()

        if self.use_pixels:
            rgb = flatten_time_dim_into_channel_dim(
                stack_tensor_dictionary(extract_many_from_batch(obs, r"rgb.*"), 1),
                has_view_axis=True,
            )

            image = rgb.float().detach()

        action = self.actor(qpos, image)

        return action

    @override
    def update(
        self, replay_iter, step: int, replay_buffer: ReplayBuffer = None
    ) -> dict:
        """
        Update the agent's policy using behavioral cloning.

        Args:
            replay_iter (iterable): An iterator over a replay buffer.
            step (int): The current step.
            replay_buffer (ReplayBuffer): The replay buffer.

        Returns:
            dict: Dictionary containing training metrics.

        """

        metrics = dict()
        batch = next(replay_iter)
        batch = {k: v.float().to(self.device) for k, v in batch.items()}

        actions = batch["action"]
        reward = batch["reward"]

        if self.low_dim_size > 0:
            obs = flatten_time_dim_into_channel_dim(
                extract_from_batch(batch, "low_dim_state")
            )
            qpos = obs.detach()

        rgb = flatten_time_dim_into_channel_dim(
            # Don't get "tp1" obs
            stack_tensor_dictionary(
                extract_many_from_batch(batch, r"rgb(?!.*?tp1)"), 1
            ),
            has_view_axis=True,
        )
        # (bs, v, c*fs, h, w)
        image = rgb.float().detach()

        task_emb = None
        if self.actor.encoder_model.use_lang_cond:
            lang_tokens = flatten_time_dim_into_channel_dim(
                extract_from_spec(batch, "lang_tokens")
            )
            task_emb, _ = self.encode_clip_text(lang_tokens)

        # If action contains all zeros, it is padded.
        is_pad = actions.sum(axis=-1) == 0
        loss_dict = self.actor(
            qpos, image, actions=actions, is_pad=is_pad, task_emb=task_emb
        )

        # calculate gradient
        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()

        # step optimizer
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
            metrics["actor_loss"] = loss_dict["loss"].item()
            metrics["actor_l1_loss"] = loss_dict["l1"].item()
            metrics["actor_kl_loss"] = loss_dict["kl"].item()
            metrics["batch_reward"] = reward.mean().item()

        return metrics

    def reset(self, step: int, agents_to_reset: list[int]):
        pass  # TODO: Implement LSTM support.

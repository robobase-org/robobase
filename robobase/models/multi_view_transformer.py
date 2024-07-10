from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from robobase.models.fusion import FusionModule
from robobase.models.act.utils.misc import kl_divergence
from robobase.models.act.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    Transformer,
)
import numpy as np
from torch.autograd import Variable


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick.

    Args:
        mu (torch.Tensor): Mean of the distribution.
        logvar (torch.Tensor): Logarithm of the variance.

    Returns:
        torch.Tensor: Reparameterized sample.
    """

    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """
    Generate a sinusoidal encoding table for positional embeddings.

    Args:
        n_position (int): Number of positions.
        d_hid (int): Hidden dimension.

    Returns:
        torch.Tensor: Sinusoidal encoding table.
    """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def build_encoder(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, pre_norm):
    d_model = hidden_dim  # 256
    dropout = dropout  # 0.1
    nhead = nheads  # 8
    dim_feedforward = dim_feedforward  # 2048
    num_encoder_layers = enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


class MultiViewTransformerEncoderDecoderACT(FusionModule):
    """
    Multi-View Transformer Encoder-Decoder for ACT model.

    Args:
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        nheads (int): Number of attention heads.
        dim_feedforward (int): Dimension of feedforward network.
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        pre_norm (bool): Use pre-normalization.
        state_dim (int): Dimension of state.
        action_dim (int): Dimension of action.
        num_queries (int): Number of queries. Equivalent to length of action sequence.
        kl_weight (int): Weight for KL divergence.
        use_lang_cond (bool): Use film for language conditioning

    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 3200,
        enc_layers: int = 4,
        dec_layers: int = 1,
        pre_norm: bool = False,
        state_dim: int = 8,
        action_dim: int = 8,
        num_queries: int = 1,
        kl_weight: int = 10,
        use_lang_cond: bool = False,
    ):
        # V, F, v is view, F is feats (token size)
        super().__init__(input_shape)

        self.dec_layers = dec_layers
        self.state_dim = state_dim
        self.num_queries = num_queries
        self.use_lang_cond = use_lang_cond

        self.encoder = build_encoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            pre_norm=pre_norm,
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=pre_norm,
            return_intermediate_dec=True,
        )

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            action_dim, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            self.state_dim, hidden_dim
        )  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.num_queries, hidden_dim),
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            3 if self.use_lang_cond else 2, hidden_dim
        )  # learned position embedding for proprio, latent and optionally text

        self.kl_weight = kl_weight

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return (
            (self.num_queries, self.state_dim),
            (self.num_queries, 1),
            (self.latent_dim),
        )

    def style_variable_encoder(
        self, bs: int, actions: torch.Tensor, qpos: torch.Tensor, is_pad: torch.Tensor
    ) -> torch.Tensor:
        """
        Style Variable Encoder for MultiViewTransformerEncoderDecoderACT model.

        Args:
            bs (int): Batch size.
            actions (torch.Tensor): Tensor containing action sequences.
            qpos (torch.Tensor): Tensor containing proprioception features.
            is_pad (torch.Tensor): Tensor indicating padding positions.

        Returns:
            torch.Tensor: Encoded style variables.
        """
        # project action sequence to embedding dim, and concat with a CLS token
        action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
        qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
        qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)

        cls_embed = self.cls_embed.weight  # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
            bs, 1, 1
        )  # (bs, 1, hidden_dim)
        encoder_input = torch.cat(
            [cls_embed, qpos_embed, action_embed], axis=1
        )  # (bs, seq+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
        # do not mask cls token
        cls_joint_is_pad = torch.full((bs, 2), False).to(
            qpos.device
        )  # False: not a padding
        is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        # query model
        encoder_output = self.encoder(
            encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
        )

        return encoder_output[0]  # take cls output only

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        qpos: torch.Tensor,
        actions: torch.Tensor = None,
        is_pad: torch.Tensor = None,
        task_emb: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the MultiViewTransformerEncoderDecoderACT model.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor]):
                    Image features and positional encodings.
            qpos (torch.Tensor): Tensor containing proprioception features.
            actions (torch.Tensor, optional): Tensor containing action sequences.
            is_pad (torch.Tensor, optional): Tensor indicating padding positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                    Tuple containing action predictions,
                    padding predictions,
                    and a list of latent variables [mu, logvar].
        """

        bs = x[0].shape[0]

        # Proprioception features
        proprio_input = self.input_proj_robot_state(qpos)

        if self.training and actions is not None:
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            # Compress action and qpos into style variable: latent_input
            encoder_output = self.style_variable_encoder(bs, actions, qpos, is_pad)

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)

        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        # Apply transformer block
        # Change to get the last output after passing through all decoder layer.
        # Fix the bug https://github.com/tonyzhaozh/act/issues/25#issue-2258740521
        hs = self.transformer(
            x[0],
            None,
            self.query_embed.weight,
            x[1],
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            task_emb=task_emb,
        )[-1]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]

    def calculate_loss(
        self,
        input_feats: Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, dict]]:
        """
        Calculate the loss for the MultiViewTransformerEncoderDecoderACT model.

        Args:
            input_feats (Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]):
                    Tuple containing action predictions, padding predictions,
                    and a list of latent variables [mu, logvar].
            actions (torch.Tensor): Tensor containing ground truth action sequences.
            is_pad (torch.Tensor): Tensor indicating padding positions.

        Returns:
            Optional[Tuple[torch.Tensor, dict]]:
                    Tuple containing the loss tensor and a dictionary of loss
                    components.
        """
        a_hat = input_feats[0]
        mu, logvar = input_feats[2]

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight

        return (loss_dict["loss"], loss_dict)

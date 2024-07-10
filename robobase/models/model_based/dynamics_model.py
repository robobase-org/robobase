from abc import ABC
from typing import Dict, Union

import torch
from torch import distributions as torchd
from torch import nn

import robobase.utils as utils
from robobase.models.model_based.utils import static_scan
from robobase.models.model_based.distributions import (
    OneHotDist,
)
from robobase.models.core import RoboBaseModule


class DynamicsModelModule(RoboBaseModule, ABC):
    def __init__(self, input_shape: tuple[int], action_dim: int):
        super().__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        assert len(input_shape) == 1, f"Expected input shape (C), but got {input_shape}"


class RSSM(DynamicsModelModule):
    def __init__(
        self,
        input_shape: tuple[int],
        action_dim: int,
        stoch: int = 32,
        discrete: int = 32,
        deter: int = 1024,
        hidden: int = 1024,
        min_std: float = 0.1,
        unimix_ratio: float = 0.01,
        initial: str = "learned",
    ):
        """Recurrent State Space Model (RSSM) introduced in PlaNet.

        Args:
            embed_dim: Dimension of the input embedding.
            action_dim: Dimension of the input action.
            stoch (optional): Size of the stochastic variable.
            discrete (optional): Size of the discrete latents.
            deter (optional): Size of deterministic latents.
            hidden (optional): Hidden size of the internal linear layers.
            min_std (optional): Minimum std for Gaussian variable.
            unimix_ratio (optional): Uniform mixture for Discrete variable.
        """
        super(RSSM, self).__init__(input_shape, action_dim)
        embed_dim = input_shape[0]
        self._stoch = stoch
        self._discrete = discrete
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._unimix_ratio = unimix_ratio
        self._initial = initial

        # RNN Modules
        self._cell = GRUCell(self._hidden, self._deter, norm=True)

        stoch_dim = stoch * discrete if discrete != 0 else stoch
        self.module_dict = nn.ModuleDict(
            {
                # Prior Modules
                "img_in": nn.Sequential(
                    nn.Linear(stoch_dim + action_dim, self._hidden, bias=False),
                    nn.LayerNorm(self._hidden, 1e-3),
                ),
                "img_out": nn.Sequential(
                    nn.Linear(deter, self._hidden, bias=False),
                    nn.LayerNorm(self._hidden, 1e-3),
                    nn.SiLU(),
                ),
                "img_dist": nn.Linear(
                    self._hidden, stoch_dim if discrete else stoch_dim * 2
                ),
                # Posterior Modules
                "obs_out": nn.Sequential(
                    nn.Linear(deter + embed_dim, self._hidden, bias=False),
                    nn.LayerNorm(self._hidden, 1e-3),
                    nn.SiLU(),
                ),
                "obs_dist": nn.Linear(
                    self._hidden, stoch_dim if discrete else stoch_dim * 2
                ),
            }
        )
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter)),
                requires_grad=True,
            )
        self.apply(utils.weight_init)
        self.module_dict["img_dist"].apply(utils.uniform_weight_init(1.0))
        self.module_dict["obs_dist"].apply(utils.uniform_weight_init(1.0))

    def initial(self, batch_size: int, device: Union[str, torch.device]):
        """Initialize RSSM deterministic & stochastic state"""
        deter = torch.zeros(batch_size, self._deter).to(device)
        if self._discrete:
            shape = [batch_size, self._stoch, self._discrete]
            state = dict(
                logit=torch.zeros(shape).to(device),
                stoch=torch.zeros(shape).to(device),
                deter=deter,
            )
        else:
            shape = [batch_size, self._stoch]
            state = dict(
                mean=torch.zeros(shape).to(device),
                std=torch.zeros(shape).to(device),
                stoch=torch.zeros(shape).to(device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def get_feat(self, state: Dict[str, torch.Tensor]):
        """Get RSSM features from deterministic & stochastic latents by concatenation.

        Args:
            state: A dictionary that contains RSSM state.
        """
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state: Dict[str, torch.Tensor]):
        """Get stochastic distribution from state.

        Args:
            state: A dictionary that contains RSSM state.
        """
        if self._discrete:
            logit = state["logit"].to(dtype=torch.float32)
            dist = torchd.Independent(
                OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean = state["mean"].to(dtype=torch.float32)
            std = state["std"].to(dtype=torch.float32)
            dist = torchd.Independent(torchd.Normal(mean, std), 1)
        return dist

    def img_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        sample: bool = True,
    ):
        """Process one-step transition with RSSM without access to input embedding.
          - deter_{t} = GRU(deter_{t-1}, stoch_{t-1}, action_{t-1})
          - stoch^{prior}_{t} ~ Linear(deter_{t})

        Args:
            prev_state: A dictionary that contains RSSM state.
            prev_action: [B, A]-shaped action tensor.
            sample: Sample from stochastic distribution if True,
                else if False, we use the mode of the distribution.

        Returns:
            post: A dictionary that contains RSSM state with posterior distribution
            prior: A dictionary that contains RSSM state with prior distribution
        """
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        # forward RNN and get determinstic latents
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self.module_dict["img_in"](x)
        x, deter = self._cell(x, [prev_state["deter"]])
        deter = deter[0]
        # get stochastic latents
        x = self.module_dict["img_out"](x)
        stats = self._suff_stats_layer("img_dist", x)
        dist = self.get_dist(stats)
        if self._discrete:
            stoch = dist.rsample() if sample else dist.mode()
        else:
            stoch = dist.rsample() if sample else dist.mode
        stoch = stoch.to(x.dtype)
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def obs_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
        sample: bool = True,
    ):
        """Process one-step transition with RSSM.
        We first extract the prior distribution with `img_step` that has no access to
        input embedding, then we extract stochastic distribution with input embedding.
          - deter_{t} = GRU(deter_{t-1}, stoch_{t-1}, action_{t-1})
          - stoch^{prior}_{t} ~ Linear(deter_{t})
          - stoch^{posterior}_{t} ~ Linear(deter_{t}, embed_{t})

        Args:
            prev_state: A dictionary that contains RSSM state.
            prev_action: [B, A]-shaped action tensor.
            embed: [B, D]-shaped input embedding tensor.
            is_first: [B,]-shaped is_first tensor.
            sample (optional): Sample from stochastic distribution if True,
                else if False, we use the mode of the distribution.

        Returns:
            post: A dictionary that contains RSSM state with posterior distribution
            prior: A dictionary that contains RSSM state with prior distribution
        """
        # Reset states if some samples within the batch are *new* samples
        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first), embed.device)
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                val = val * (1.0 - is_first_r) + init_state[key] * is_first_r

        # get prior discrete & stochastic latents
        prior = self.img_step(prev_state, prev_action, sample)

        # get posterior stochastic latents with access to embed
        x = torch.cat([prior["deter"], embed], -1)
        x = self.module_dict["obs_out"](x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        if self._discrete:
            stoch = dist.rsample() if sample else dist.mode()
        else:
            stoch = dist.rsample() if sample else dist.mode
        stoch = stoch.to(x.dtype)
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def get_stoch(self, deter):
        x = self.module_dict["img_out"](deter)
        stats = self._suff_stats_layer("img_dist", x)
        dist = self.get_dist(stats)
        if self._discrete:
            out = dist.mode()
        else:
            out = dist.mode
        return out

    def _suff_stats_layer(self, name: str, x: torch.Tensor):
        """Extract sufficient statistics required for the stochastic distribution.
        For Gaussian, outputs are mean and standard deviation.
        For Discrete, outputs are logits.

        Args:
            name: string key that denotes the linear layer used for this function
                - "obs_dist" for posterior distribution
                - "img_dist" for prior distribution
            x: [B, T, D] or [B, D] shaped tensor
        """
        x = self.module_dict[name](x)
        if self._discrete != 0:
            logit = x.reshape([*x.shape[:-1], self._stoch, self._discrete])
            return {"logit": logit}
        else:
            mean, std = torch.split(x, self._stoch, -1)
            std = 2 * torch.sigmoid(std / 2)
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor,
        state: Dict[str, torch.Tensor] = None,
    ):
        """Processes a sub-trajectory through RSSM.
        Not all transitions does not have to be from the same episode,
        because the model will initialize a new state when encountering is_first.

        Args:
            embed: [B, T, D]-shaped input embedding tensor
            action: [B, T, A]-shaped action tensor
            is_first: [B, T]-shaped is_first tensor
            state: A dictionary that contains RSSM state.
                If this is set to None, we initialize a new one
        """

        def swap(x: torch.Tensor):
            return x.permute([1, 0] + list(range(2, len(x.shape))))

        if state is None:
            state = self.initial(action.shape[0], embed.device)
        # (B, T, ch) -> (T, B, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of (posterior, prior) from obs_step
        post, prior = static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )
        # (T, B, ch) -> (B, T, ch)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action: torch.Tensor, state: Dict[str, torch.Tensor] = None):
        """Processes an action sequence through RSSM.
        This corresponds to predicting future latents conditioned on future actions.
        This method is mostly used for evaluation purposes (e.g., visualization)

        Args:
            action: [B, T, A]-shaped action tensor
            state: A dictionary that contains RSSM state.
                If this is set to None, we initialize a new one
        """

        def swap(x):
            return x.permute([1, 0] + list(range(2, len(x.shape))))

        if state is None:
            state = self.initial(action.shape[0], action.device)
        assert isinstance(state, dict), state
        action = swap(action)
        prior = static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def kl_loss(self, post: dict, prior: dict):
        """Computes the KL loss objective for training RSSM
        - Representation loss (rep) encourages posterior to be close to prior,
            which makes s_{t+1} to be in meaningful representation space.
        - Dynamics loss (dyn) encourages prior to be close to posterior,
            which means predicting the future latent s_{t+1} without access to o_{t+1}.

        Args:
            post: A dictionary that contains Posterior RSSM state.
            prior: A dictionary that contains Prior RSSM state.
        """

        def dist(x: dict):
            return self.get_dist(x)

        def sg(x: dict):
            return {k: v.detach() for k, v in x.items()}

        kld = torchd.kl.kl_divergence
        # Representation loss: Post --> Prior
        rep_loss = value = kld(dist(post), dist(sg(prior)))
        # Dynamics loss: Prior --> Post
        dyn_loss = kld(dist(sg(post)), dist(prior))
        return rep_loss, dyn_loss, value

    @property
    def output_shape(self):
        out_dim = self._deter
        if self._discrete != 0:
            out_dim += self._stoch * self._discrete
        else:
            out_dim += self._stoch
        return (out_dim,)


class GRUCell(nn.Module):
    def __init__(
        self,
        inp_size,
        size,
        norm=False,
        act=torch.tanh,
        update_bias=-1,
    ):
        """
        GRU class that supports normalization

        Args:
            inp_size (int): Dimension of the inputs.
            size (int): Dimension of the GRU latents.
            norm (bool, optional): Whether to use layer normalization.
            act (function, optional): Activation of the GRU latents.
            update_bias (int, optional): Bias term for GRU updates.
        """
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

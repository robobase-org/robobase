from typing import Union, Callable, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.distributions as torchd

from robobase.models.fully_connected import FullyConnectedModule
import robobase.models.model_based.distributions as distributions


class RequiresGrad:
    def __init__(self, model: Union[nn.Module, List[nn.Module]]):
        if isinstance(model, nn.Module):
            model = [model]
        self._model = model

    def __enter__(self):
        for model in self._model:
            model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        for model in self._model:
            model.requires_grad_(requires_grad=False)


def flatten_nested(structure: Union[list, tuple, dict, torch.Tensor]):
    """Flatten a nested structure (lists/tuples/dicts) into a flat list."""
    flat_list = []

    if isinstance(structure, (list, tuple)):
        for item in structure:
            flat_list.extend(flatten_nested(item))
    elif isinstance(structure, dict):
        for key, value in structure.items():
            flat_list.extend(flatten_nested(value))
    else:
        flat_list.append(structure)

    return flat_list


def pack_sequence_as(
    structure: Union[list, tuple, dict, torch.Tensor],
    flat_sequence: Union[list, tuple, dict, torch.Tensor],
):
    """Packs a flat sequence into a given structure."""
    packed, remaining = pack_sequence_as_helper(structure, flat_sequence)
    if len(remaining) != 0:
        raise ValueError("Flat sequence has more elements than the structure.")
    return packed


def pack_sequence_as_helper(
    structure: Union[list, tuple, dict, torch.Tensor],
    flat_sequence: Union[list, tuple, dict, torch.Tensor],
):
    """Helper function for pack_sequence_as."""
    if isinstance(structure, (list, tuple)):
        packed_elements = []
        for item in structure:
            packed, flat_sequence = pack_sequence_as_helper(item, flat_sequence)
            packed_elements.append(packed)
        return type(structure)(packed_elements), flat_sequence
    elif isinstance(structure, dict):
        packed_dict = {}
        for key, value in structure.items():
            packed, flat_sequence = pack_sequence_as_helper(value, flat_sequence)
            packed_dict[key] = packed
        return packed_dict, flat_sequence
    else:
        if not flat_sequence:
            raise ValueError("Flat sequence has fewer elements than the structure.")
        return flat_sequence.pop(0), flat_sequence


def reorder_dict_keys(
    target: Union[list, tuple, dict, torch.Tensor],
    template: Union[list, tuple, dict, torch.Tensor],
):
    """Recursively reorder the keys of dictionaries in the target to match
    the key order in corresponding dictionaries in template.
    """
    if isinstance(template, dict) and isinstance(target, dict):
        # If both template and target are dictionaries, reorder the keys in target
        return {
            k: reorder_dict_keys(target.get(k, None), v) for k, v in template.items()
        }
    elif isinstance(template, (list, tuple)) and isinstance(target, (list, tuple)):
        # If both are lists/tuples, recursively process their elements
        return type(template)(
            [reorder_dict_keys(t, s) for t, s in zip(target, template)]
        )
    else:
        # If neither is a dict or list/tuple, return the target unchanged
        return target


def static_scan(
    fn: Callable,
    inputs: Tuple[torch.Tensor],
    start: Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]],
    reverse: bool = False,
):
    """Sequentially applies a function to inputs from start values,
    which could be useful for computing value functions or imagining future states.
    NOTE: Not actual scan functionality unlike Jax & TF, as torch does not support it

    Args:
        fn: A function that will be applied to inputs.
        inputs: inputs that will be given to the function.
            each element in tuple will be in [T, ...]-shaped, and
            (inputs[0][i], inputs[1][i], ..) will be given to the function.
        start: Start values for the function.
            For instance, this could be the RSSM state dictionary of the first state
            ({"deter": [B, D], "logit": , ...} that will be used for future imagination
        reverse (optional): Reverses the sequence. This could be useful for
            lambda_return computation that recursively does bootstrapping from backward

    Returns:
        tuple/dictionary of torch.Tensor with the same data structure as `start`
        e.g.) ({"deter": [T, B, D], "logit": , ...}) for future imagination
    """
    last = start
    outputs = [[] for _ in flatten_nested(start)]
    indices = range(flatten_nested(inputs)[0].shape[0])
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = (_input[index] for _input in inputs)
        last = fn(last, *inp)
        last = reorder_dict_keys(last, start)  # make the order of data structure same
        [_out.append(_last) for _out, _last in zip(outputs, flatten_nested(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [torch.stack(x, 0) for x in outputs]
    return pack_sequence_as(start, outputs)


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    pcont: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
    axis: int,
):
    """Computes lambda_return for a given trajectory.
    - Setting lambda=1 gives a discounted Monte Carlo return.
    - Setting lambda=0 gives a fixed 1-step return.

    Args (Assuming axis == 0):
        axis: axis corresponding to the temporal dimension
        reward: [T, B]-shaped reward tensor
        value: [T, B]-shaped value tensor from critic
        pcont: [T, B]-shaped pcont tensor which denotes the probability
            of next state not being the terminal state (probability of continuation)
        bootstrap: [B,]-shape last-state value tensor from critic,
            which will be used for bootstrapping.
        lambda_: Lambda hyperparameter for computing the lambda return

    Returns (Assuming axis == 0):
        [T, B]-shaped lambda return tensor
    """
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)

    returns = static_scan(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
        (inputs, pcont),
        bootstrap,
        reverse=True,
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class ReturnEMA(nn.Module):
    """
    A class for normalizing PyTorch Tensor with running 5%, 95% percentile stats
    """

    def __init__(self, alpha: float = 1e-2):
        super().__init__()
        self.values = nn.Parameter(torch.zeros((2,)), requires_grad=False)
        self.range = nn.Parameter(torch.tensor([0.05, 0.95], requires_grad=False))
        self.alpha = alpha

    def __call__(self, x: torch.Tensor):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values.data.copy_(
            self.alpha * x_quantile + (1 - self.alpha) * self.values.data
        )
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class BatchTimeInputDictWrapperModule(nn.Module):
    """Transforms tensor inputs to dictionary given specified key.

    This wrapper takes [B, T, ..] shape input, reshape it to [B * T, ..],
    wrap it as a dictionary with given key, and reshapes output to original shape.

    Useful as a wrapper for RoboBase's FullyConnectedModule when we assume that
    there is only one Tensor input to FullyConnectedModule
    """

    def __init__(self, module: FullyConnectedModule, key: str):
        super().__init__()
        self.module = module
        self.key = key

    def forward(self, x: torch.Tensor):
        init_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        out = self.module({self.key: x})
        out = out.view(*init_shape, *out.shape[1:])
        return out

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped module, if attribute
        is not found in this wrapper class.
        """
        try:
            # Try to get attribute from the parent class first
            return super().__getattr__(name)
        except AttributeError:
            # If it is not found, forward the request to the wrapped module
            return getattr(self.module, name)


class DistWrapperModule(nn.Module):
    def __init__(self, module: nn.Module, dist: str):
        super().__init__()
        self.module = module
        self._dist = dist

    def forward(self, inputs: torch.Tensor):
        x = self.module(inputs)
        dist = self.get_dist(x)
        if self._dist == "binary":
            return dist.mean()
        else:
            return dist.mode()

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        x = self.module(inputs)
        dist = self.get_dist(x)
        loss = -dist.log_prob(targets)
        return loss

    def get_dist(self, x):
        if self._dist == "binary":
            x = x.squeeze(-1)
            dist = distributions.Bernoulli(
                torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=x), 0)
            )
        elif self._dist == "symlog_disc":
            dist = distributions.DiscSymLogDist(x)
        else:
            raise ValueError(self._dist)
        return dist

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped module, if attribute
        is not found in this wrapper class.
        """
        try:
            # Try to get attribute from the parent class first
            return super().__getattr__(name)
        except AttributeError:
            # If it is not found, forward the request to the wrapped module
            return getattr(self.module, name)


def batch_time_forward(
    fn: Union[Callable, nn.Module],
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
):
    reshaped_inputs, original_shapes = [], []
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    for input_tensor in inputs:
        original_shapes.append(input_tensor.shape)
        reshaped_input = input_tensor.view(-1, *input_tensor.shape[2:])
        reshaped_inputs.append(reshaped_input)

    outputs = fn(*reshaped_inputs)
    reshaped_outputs = outputs.view(*original_shapes[0][:2], *outputs.shape[1:])
    return reshaped_outputs

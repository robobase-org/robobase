import torch
import torch.nn.functional as F


from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough
from torch.distributions import Distribution


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class OneHotDist(OneHotCategoricalStraightThrough):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        """Wrapper for torch.distribution.OneHotCategorical.
        - This class supports uniform mixture following DreamerV3 implementation
        - This class supports dist.mode have a straight-through gradient
        """
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()


class Bernoulli:
    def __init__(self, dist: Distribution):
        """Wrapper for torch.distribution.Bernoulli.
        - This class supports straight-through gradient estimator
        - Makes the log_prob work the same way as in TensorFlow implementation
        """
        super().__init__()
        self._dist = dist
        self._mean = dist.mean

    def __getattr__(self, name: str):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def mean(self):
        return self._mean

    def sample(self, sample_shape: tuple = ()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x: torch.Tensor):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class TruncatedNormalWithScaling(torch.distributions.normal.Normal):
    def __init__(self, loc, scale, absmax=1):
        super().__init__(loc, scale)
        self._absmax = absmax

    def mode(self):
        out = self.loc
        if self._absmax is not None:
            out *= (
                self._absmax / torch.clip(torch.abs(out), min=self._absmax)
            ).detach()
        return out

    def sample(self, sample_shape):
        out = super().sample(sample_shape)
        if self._absmax is not None:
            out *= (
                self._absmax / torch.clip(torch.abs(out), min=self._absmax)
            ).detach()
        return out

    def rsample(self, sample_shape):
        out = super().rsample(sample_shape)
        if self._absmax is not None:
            out *= (
                self._absmax / torch.clip(torch.abs(out), min=self._absmax)
            ).detach()
        return out


class DiscSymLogDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(logits.device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients
        # are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)

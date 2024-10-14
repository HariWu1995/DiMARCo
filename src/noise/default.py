"""
For more advanced schedulers:
    https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers
"""
import math
import torch

from ..const import eps


def _schedule(x, scale):
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale)
    scale = scale.view(-1, *[1] * len(x.shape[1:]))
    noise = torch.rand_like(x.to(torch.float))
    return noise, scale


def alpha_schedule(x, alpha: float or torch.Tensor = 0.49):
    noise, alpha = _schedule(x, apha)
    return (1 - alpha) * x + alpha * noise


def beta_schedule(x, beta: float or torch.Tensor = 0.49):
    noise, beta = _schedule(x, beta)
    return torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise



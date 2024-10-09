"""
For more advanced schedulers:
    https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers
"""
import math
import torch

from ..const import eps


def alpha_schedule(x, alpha: float = 0.1):
    noise = torch.rand_like(x)
    return (1 - alpha) * x + alpha * noise


def beta_schedule(x, beta: float = 0.1):
    noise = torch.randn_like(x)
    return torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise



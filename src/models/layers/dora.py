"""
Weight-Decomposed Low-Rank Adaptation

Reference:
    https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
    https://github.com/catid/dora/blob/main/dora.py
"""
import torch
from torch import nn
from torch.nn import functional as F

from .lora import LoRA


class LinearDoRA(nn.Module):

    def __init__(self, linear, rank: int = 2, weight: float = 1.):
        super().__init__()
        assert isinstance(linear, nn.Linear), \
                        "LinearDoRA only accepts torch.nn.Linear as 1st input"
        self.linear = linear
        self.adapter = LoRA(linear.in_features, linear.out_features, rank, weight)
        self.m = nn.Parameter(linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):

        # LoRA Calculation
        adapter = self.adapter.w * (self.adapter.A @ self.adapter.B).T
        weight_l = self.linear.weight + adapter

        # DoRA Calculation
        direction = weight_l / weight_l.norm(p=2, dim=0, keepdim=True)
        weight_d = self.m * direction

        return F.linear(x, weight_d, self.linear.bias)


if __name__ == "__main__":

    torch.manual_seed(123)
    x = torch.randn((1, 10))

    layer = nn.Linear(10, 2)
    print("Linear output:", layer(x))

    layer_dora = LinearDoRA(layer, rank=2, weight=1.2345)
    print("DoRA output:", layer_dora(x))


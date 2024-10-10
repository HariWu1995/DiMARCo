"""
Low-Rank Adaptation

Reference:
    https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
    https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
"""
import torch
from torch import nn


class LoRA(nn.Module):

    def __init__(self, in_dim, out_dim, rank: int = 2, weight: float = 1.):
        
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.w = weight

    def forward(self, x):
        x = self.w * (x @ self.A @ self.B)
        return x


class LinearLoRA(nn.Module):

    def __init__(self, linear, rank: int = 2, weight: float = 1.):
        super().__init__()
        assert isinstance(linear, nn.Linear), \
                        "LinearLoRA only accepts torch.nn.Linear as 1st input"
        self.linear = linear
        self.adapter = LoRA(linear.in_features, linear.out_features, rank, weight)

    def forward(self, x):
        return self.linear(x) + self.adapter(x)


if __name__ == "__main__":

    torch.manual_seed(123)
    x = torch.randn((1, 10))

    layer = nn.Linear(10, 2)
    print("Linear output:", layer(x))

    layer_lora = LinearLoRA(layer, rank=2, weight=1.2345)
    print("LoRA output:", layer_lora(x))


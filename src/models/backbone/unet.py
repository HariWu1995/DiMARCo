"""
Reference: 
    https://huggingface.co/docs/diffusers/en/api/models/unet2d
    https://huggingface.co/learn/diffusion-course/unit1/3#the-unet
"""
from diffusers import UNet2DModel as UNet2D

import torch
from torch import nn


class UNet(nn.Module):

    def __init__(self, in_channels: int =  1, 
                      out_channels: int = -1,
                        num_stages: int =  2, **kwargs):
        
        super().__init__()

        self.num_stages = num_stages

        if out_channels <= 0:
            out_channels = in_channels

        block_channels = [2**(5+i) for i in range(num_stages)]
        block_channels += [block_channels[-1]]
        block_channels = tuple(block_channels)

        num_resnet_layers = kwargs.get('num_resnet_layers', 2)
        down_block_layers = ("DownBlock2D","AttnDownBlock2D","AttnDownBlock2D")
        up_block_layers = ("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D")

        self.backbone = UNet2D(
                            in_channels = in_channels,
                           out_channels = out_channels,
                     block_out_channels = block_channels,
                       layers_per_block = num_resnet_layers,
                       down_block_types = down_block_layers,
                         up_block_types =   up_block_layers,
                    )

    def forward(self, x, t: int = 1):

        N = self.num_stages
        gcd = 2 ** N

        H, W = x.shape[-2:]
        assert (H % gcd == 0) \
           and (W % gcd == 0), f"input_size ({H}, {W}) must be divisible by {gcd}"

        y = self.backbone(x, t, return_dict=False)[0]
        return y


if __name__ == "__main__":

    model = UNet()
    # print(model)
    print(sum([p.numel() for p in model.parameters()]))

    # W, H must be divisible by 2 ** (num_stages)
    x = torch.rand(10, 1, 28, 28)
    print(x.shape)

    y = model(x)
    print(y.shape)


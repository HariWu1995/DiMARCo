"""
Reference: 
    https://huggingface.co/docs/diffusers/en/api/models/unet2d
    https://huggingface.co/learn/diffusion-course/unit2/3#making-a-class-conditioned-diffusion-model
"""
from diffusers import UNet2DModel as UNet2D

import math
import torch
from torch import nn


class CatUNet(nn.Module):
    """
    Categorical U-Net
    """
    def __init__(self, in_channels: int =  1, 
                      out_channels: int = -1,
                      init_filters: int = 32, 
                        num_stages: int =  2, 
                       num_classes: int = 10, 
                  background_class: int = -1, **kwargs):
        
        super().__init__()

        self.num_stages = num_stages
        self.num_classes = num_classes + (1 if isinstance(background_class, int) else 0)
        self.background_class = background_class
        
        embedding_size = kwargs.get('embedding_size', math.ceil(math.log2(num_classes)))
        padding_class = 0 if background_class is not None else None

        self.embedder = nn.Embedding(self.num_classes, embedding_size, padding_idx=padding_class)
        # print(self.embedder.weight)

        if out_channels <= 0:
            out_channels = in_channels

        block_channels = [init_filters * (2**i) for i in range(num_stages)]
        block_channels += [block_channels[-1]]
        block_channels = tuple(block_channels)

        num_resnet_layers = kwargs.get('num_resnet_layers', 2)
        down_block_layers = ("DownBlock2D","AttnDownBlock2D","AttnDownBlock2D")
        up_block_layers = ("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D")

        self.backbone = UNet2D(
                            in_channels = in_channels + embedding_size,
                           out_channels = out_channels,
                     block_out_channels = block_channels,
                       layers_per_block = num_resnet_layers,
                       down_block_types = down_block_layers,
                         up_block_types =   up_block_layers,
                    )

    def forward(self, x, c = None, t: int = 1):

        N = self.num_stages
        gcd = 2 ** N

        H, W = x.shape[-2:]
        assert (H % gcd == 0) \
           and (W % gcd == 0), f"input_size ({H}, {W}) must be divisible by {gcd}"

        bg = self.background_class if self.background_class else 0

        # In case x is categorical value
        if c is None:
            c = x
            x = torch.where(x > bg, 1., 0)

        # Assume that `background_class` is the minimum among all classes
        c = c - bg
        c = self.embedder(c)
        c = torch.swapaxes(c, 1, 4).squeeze(dim=-1)
            
        # Concat -> (B, 1 + num_classes, H, W)
        x = torch.cat([x, c], dim=1)

        y = self.backbone(x, t, return_dict=False)[0]
        return y


if __name__ == "__main__":

    model = CatUNet(num_classes=10, background_class=-1)
    # print(model)
    print(sum([p.numel() for p in model.parameters()]))

    # W, H must be divisible by 2 ** (num_stages)
    x = torch.randint(low=-1, high=10, size=(10, 1, 28, 28))
    print(x.shape)

    y = model(x)
    print(y.shape)


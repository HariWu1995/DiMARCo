"""
Minimal U-Net (µnet)
    https://huggingface.co/learn/diffusion-course/unit1/3
"""
import torch
from torch import nn


class mUNet(nn.Module):

    def __init__(self, in_channels: int =  1, 
                      out_channels: int = -1,
                       kernel_size: int =  5,
                      padding_size: int =  2,
                     dilation_size: int =  1,
                        num_stages: int =  2, **kwargs):
        
        super().__init__()
        
        self.num_stages = num_stages

        si = in_channels
        so = out_channels if out_channels > 0 else in_channels
        
        self.scalers = nn.ModuleDict(dict(
                  up = nn.Upsample(scale_factor=2),
                down = nn.MaxPool2d(kernel_size=2),
            ))

        self.activator = nn.PReLU() if kwargs.get('learnable_activation', False) \
                    else nn.SiLU()

        ## Feature Extraction: start with 2 ** 5 = 32
        conv_kwargs = dict( kernel_size = kernel_size, 
                                padding = padding_size, 
                               dilation = dilation_size, )

        self.featractors = nn.ModuleDict(dict(
                    down = nn.ModuleList([nn.Conv2d(si, 2**5, **conv_kwargs)]),
                      up = nn.ModuleList([nn.Conv2d(2**5, so, **conv_kwargs)]),
                ))

        for i in range(5, 5+num_stages-1):
            self.featractors['down'].append(nn.Conv2d(2**i, 2**(i+1), **conv_kwargs))
            self.featractors[ 'up' ].append(nn.Conv2d(2**(i+1), 2**i, **conv_kwargs))

        self.featractors['down'].append(nn.Conv2d(2**(i+1), 2**(i+1), **conv_kwargs))
        self.featractors[ 'up' ].append(nn.Conv2d(2**(i+1), 2**(i+1), **conv_kwargs))

        self.featractors['up'] = self.featractors['up'][::-1]

    def forward(self, x):

        assert len(self.featractors['up']) == len(self.featractors['down'])

        N = self.num_stages
        gcd = 2 ** N

        H, W = x.shape[-2:]
        assert (H % gcd == 0) \
           and (W % gcd == 0), f"input_size ({H}, {W}) must be divisible by {gcd}"

        temp = []

        for di, downer in enumerate(self.featractors['down']):
            x = downer(x)
            if di < N:
                temp.append(x)
                x = self.scalers['down'](x)

        for ui, upper in enumerate(self.featractors['up']):
            if ui >= 1:
                x = self.scalers['up'](x)
                x += temp.pop()
            x = upper(x)

        x = self.activator(x)
        return x


if __name__ == "__main__":

    model = mUNet()
    # print(model)
    print(sum([p.numel() for p in model.parameters()]))

    # W, H must be divisible by 2 ** (num_stages)
    x = torch.rand(10, 1, 28, 28)
    print(x.shape)

    y = model(x)
    print(y.shape)



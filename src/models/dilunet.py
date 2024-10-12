import torch
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d as ConvTr2d, BatchNorm2d as Norm2d
from torch.nn import functional as F


class DiConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DiConv, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            Norm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            Norm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    """
    Downscaling with max-pool then double-conv
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DownConv, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DiConv(in_channels, out_channels, dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear: bool = True):
        super(UpConv, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DiConv(in_channels, out_channels, dilation=1)
        else:
            self.up = ConvTr2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
            self.conv = DiConv(in_channels, out_channels, dilation=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW (channels, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate along the channel axis
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(OutConv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DilUNet(nn.Module):
    """
    Dilated-Conv U-Net
    """
    def __init__(self, in_channels: int =  1, 
                      out_channels: int = -1,
                        num_stages: int =  2, 
                      init_filters: int = 32, 
                  upscale_bilinear: bool = True, **kwargs):
        
        super().__init__()
        
        self.num_stages = num_stages

        self.activator = nn.PReLU() if kwargs.get('learnable_activation', False) \
                    else nn.SiLU()

        ## Feature Extraction
        f = init_filters
        si = in_channels
        so = out_channels if out_channels > 0 else in_channels

        dilations = [2**i for i in range(num_stages)]
        filters = [f*(2**i) for i in range(num_stages)]

        self.featract_in = DiConv(si, filters[0], dilation=1)
        self.featract_out = OutConv(filters[0], so)
        
        self.featractors = nn.ModuleDict(dict(
                    down = nn.ModuleList([DownConv(filters[i], filters[i+1], dilation=dilations[i+1]) for i in range(num_stages-1)]),
                      up = nn.ModuleList([UpConv(2*filters[i], filters[i-1], upscale_bilinear) for i in range(num_stages-1, 0, -1)]),
                    peer = DownConv(filters[-1], filters[-1], dilation=dilations[-1]),
                ))

        self.featractors['up'].append(UpConv(filters[1], filters[0], upscale_bilinear))

    def forward(self, x, **kwargs):

        N = self.num_stages
        gcd = 2 ** N

        H, W = x.shape[-2:]
        assert (H % gcd == 0) \
           and (W % gcd == 0), f"input_size ({H}, {W}) must be divisible by {gcd}"

        temp = [self.featract_in(x)]
        for di, downer in enumerate(self.featractors['down']):
            d = downer(temp[-1])
            temp.append(d)

        # Peer-to-peer
        x = temp[-1]
        x = self.featractors['peer'](x)
        
        for ui, upper in enumerate(self.featractors['up']):
            x = upper(x, temp.pop())

        x = self.featract_out(x)
        x = self.activator(x)
        return x


if __name__ == "__main__":

    model = DilUNet(num_stages=2)
    # print(model)
    # print(sum([p.numel() for p in model.parameters()]))

    # W, H must be divisible by 2 ** (num_stages)
    x = torch.rand(10, 1, 32, 32)
    print(x.shape)

    y = model(x)
    print(y.shape)



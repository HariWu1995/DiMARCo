import torch
from torch import nn


def GridConv(in_channels: int, out_channels: int, kernel_size: int, 
                  stride: int = 1, dilation: int = 1, padding: int = 0):

    return nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, dilation=dilation, padding=padding, padding_mode='zeros'),
            nn.PReLU(),
        ])    


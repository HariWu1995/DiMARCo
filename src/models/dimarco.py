"""
Diffusion-Model for ARCorpus
DI    -   M     -   ARCO    

Reference:
    https://huggingface.co/learn/diffusion-course/en/unit1/3
"""
import math

import torch
from torch import nn

from .layers import GridConv2d
from ..const import eps


class DiMARCo(nn.Module):

    def __init__(
        self, 

        # U-Net
        in_channels: int,
        out_channels: int = -1,
        
        # Diffusion
        objective: str = 'v-space', # v-space / noise
        diff_steps: int = 10, 
        min_noise: float = math.sqrt(eps), 
        max_noise: float = math.sqrt(
                           math.sqrt(eps)),
        
    ):
        ## Diffusion
        self.objective = objective
        self.beta_at_steps = torch.linspace(min_noise, max_noise, diff_steps)

        ## U-Net
        si = in_channels
        so = out_channels if out_channels > 0 else in_channels

        # Flexible Feature-Extraction for different grid-size(s)
        self.grid_conv = nn.ModuleDict({
                 'tiny': GridConv2d(1, 16, kernel_size=3, padding=1),
                'small': GridConv2d(1, 16, kernel_size=3, padding=1),
               'medium': GridConv2d(1, 16, kernel_size=3, padding=1),
                'large': GridConv2d(1, 16, kernel_size=3, padding=1),
            })
       
        # Define the U-Net stages with reduced convolutions
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder part (upsampling layers)
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.decoder3 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.decoder4 = nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)  # Output to 1 channel
    
    def forward(self, x):
        input_size = x.size(2)  # assuming square input, so width == height
        
        # Select the appropriate stages based on input size
        if input_size > 24:


        if input_size > 24:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.decoder1(x)
            x = self.decoder2(x)
            x = self.decoder3(x)
            x = self.decoder4(x)
        
        elif input_size > 11:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.decoder2(x)
            x = self.decoder3(x)
            x = self.decoder4(x)
        
        elif input_size > 6:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.decoder3(x)
            x = self.decoder4(x)
        
        else:
            x = self.stage1(x)
            x = self.decoder4(x)
        
        return x



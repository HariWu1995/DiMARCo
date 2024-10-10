"""
Diffusion-Model for ARCorpus
DI    -   M     -   ARCO    

Reference:
    https://huggingface.co/learn/diffusion-course/en/unit1/3
"""
import math
import torch

from .backbone import UNet, CatUNet, DilUNet
from ..const import eps


class DiMARCo:

    def __init__(
        self, 

        # U-Net
        backbone: str = 'unet',
        num_stages: int = 2, 
        num_classes: int = 10, 
        background_class: int = -1,
        upscale_bilinear: bool = True,

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

        ## Backbone
        if backbone.lower() == 'catunet':
            from .backbone import CatUNet as UNet
        elif backbone.lower() == 'dilunet':
            from .backbone import DilUNet as UNet
        else:
            from .backbone import UNet
        
    def __call__(self, x):
        input_size = x.size(2)  # assuming square input, so width == height
        
        



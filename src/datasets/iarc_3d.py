"""
DataLoader to enrich data
    from naïve ARC 2D-format (H x W)
    to multi-layer 3D-format (H x W x C), 
        where C = 10 is number of non-background values
"""
from typing import List, Tuple, Union, Callable

import random as rd
import numpy as np

import torch
from torch.utils.data import Dataset

from src.arckit.data import Task, TaskSet

from .utils.encoder import encode_1hot
from .iarc_2d import iARCDatasetAug, AUGMENTATIONS


class iARCDatasetDepth(iARCDatasetAug):
    
    def __init__(
            self, 
            task_set: TaskSet = None, 
            batch_size: int = None, 
            grid_size: Union[None, int, List[int]] = None, 
            augmentors: Union[None, Callable, List, Tuple] = AUGMENTATIONS, 
          **kwargs
        ):

        super().__init__(task_set, batch_size, grid_size, augmentors, **kwargs)
    
        self.ignore_background = kwargs.get('ignore_background', False)

    def __getitem__(self, idx):

        if idx >= len(self):
            print(f'[Warning] index {idx} should be lower than {len(self)}. '
                  f'It will be reduced to {idx % len(self)}')
            idx = idx % len(self)
            
        task = self.task_set[idx]
    
        # Get padded grids -> (batch_size, batch_max_height, batch_max_width)
        grids_pad = self.extract_grid(task)[0]
        grids_pad = grids_pad.detach().cpu().numpy()

        # Augmentation
        grids_aug = self.augment(grids_pad)
        grids_aug = torch.tensor(grids_aug)

        # Extract mask
        grids_mask = torch.where(grids_aug >= 0, 1., 0.)
        grids_mask = grids_mask.unsqueeze(dim=1).repeat(1, self.num_classes, 1, 1)

        # 1-hot Encoding
        grids_3d = encode_1hot(grids_aug, num_classes = self.num_classes,
                                    ignore_background = self.ignore_background)

        # Reshape
        grids_3d = torch.swapaxes(grids_3d, 1, 3)
        grids_3d = torch.swapaxes(grids_3d, 2, 3)

        return [grids_3d, grids_mask], grids_3d


if __name__ == "__main__":
    pass



"""
DataLoader for naïve ARC 2D-format
"""
from typing import List, Tuple, Union, Callable

import random as rd
import numpy as np
import torch

from src.arckit.data import Task

from .base import ARCBaseDataset as Dataset
from .utils import encode_1hot, scale_up


class ARCDatasetNaive(Dataset):
    """
    ARC Dataset
    """
    def __init__(
            self, 
            task_set: Task = None, 
            grid_size: Union[None, int, List[int]] = None, 
          **kwargs
        ):
        super().__init__(task_set, grid_size, **kwargs)

    def __len__(self):
        return 1

    def __getitem__(self, mode: str = 'train'):

        grids_raw = self.task_set.train if mode == 'train' else \
                    self.task_set.test
        grids_in, \
        grids_out = self.extract_grid(grids_raw)

        return grids_in, grids_out

    def extract_grid(self, grids):

        # un-Parsing
        grid_max_H = 0
        grid_max_W = 0
        
        grids_raw = []
        for Gin, Gout in grids:
            grids_raw.append([Gin, Gout])
            max_h = max(Gin.shape[0], Gout.shape[0])
            max_w = max(Gin.shape[1], Gout.shape[1])

            if grid_max_H < max_h:
                grid_max_H = max_h

            if grid_max_W < max_w:
                grid_max_W = max_w

        if isinstance(self.grid_size, (list, tuple)):
            grid_size = self.grid_size
        else:
            grid_size = (grid_max_H, grid_max_W)
            if self.grid_square:
                grid_size = tuple([max(*grid_size)] * 2)

        # Padding
        grids_in = []
        grids_out = []
        for Gin, Gout in grids_raw:
            grids_in.append(self._pad(Gin, grid_size))
            grids_out.append(self._pad(Gout, grid_size))

        # Formatting
        grids_in = torch.tensor(grids_in)
        grids_out = torch.tensor(grids_out)

        return grids_in, grids_out

    def _pad(self, grid, grid_size):
        grid_upscale = max(*grid_size) // np.max(grid.shape)
        if (grid_upscale > 2) and self.normalize_size:
            grid = scale_up(grid, h_scale=grid_upscale, v_scale=grid_upscale)
        if (grid.shape != grid_size):
            grid = self.padding(grid, grid_size=grid_size)
        return grid

    def shuffle(self):
        rd.shuffle(self.task_set.train)


class ARCDatasetDepth(ARCDatasetNaive):

    def __getitem__(self, idx: int = 0):

        grids_raw = self.task_set.train if mode == 'train' else \
                    self.task_set.test
        grids_in, \
        grids_out = self.extract_grid(grids_raw)

        # Extract mask
        grids_mask = torch.where(grids_in >= 0, 1., 0.)
        grids_mask = grids_mask.unsqueeze(dim=1).repeat(1, self.num_classes, 1, 1)

        # 1-hot Encoding
        grids_in = encode_1hot(grids_in, num_classes = self.num_classes,
                                    ignore_background = self.ignore_background)
        grids_out = encode_1hot(grids_out, num_classes = self.num_classes,
                                    ignore_background = self.ignore_background)

        # Reshape
        grids_in = torch.swapaxes(grids_in, 1, 3)
        grids_out = torch.swapaxes(grids_out, 2, 3)

        return grids_in, grids_out, grids_mask



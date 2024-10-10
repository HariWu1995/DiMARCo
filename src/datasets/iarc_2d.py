"""
DataLoader for naïve ARC 2D-format
"""
from typing import List, Tuple, Union, Callable
from itertools import product as iter_product

import random as rd
import numpy as np

import torch
from torch.utils.data import Dataset

from src.arckit.data import Task, TaskSet

from .utils.augmentations import (
    self_identity, scale_up,
    rotate_90, rotate_180, rotate_270,
    flip_vertical, flip_horizontal, flip_channel,
)


# 2-stage Augmentations: 1 -> 4 -> 16
AUGMENTATIONS = [
    (self_identity, flip_vertical, flip_horizontal, flip_channel),
    (self_identity, rotate_90, rotate_180, rotate_270)
]


class iARCDatasetNaive(Dataset):
    """
    Identity ARC Dataset
    """
    def __init__(
            self, 
            task_set: TaskSet = None, 
            batch_size: int = None, 
            grid_size: Union[None, int, List[int]] = None, 
          **kwargs
        ):

        self.task_set = task_set
        self.batch_size = batch_size if batch_size else -1
        self.grid_size = [grid_size] * 2 if isinstance(grid_size, int) else grid_size
        self.grid_square = kwargs.get('grid_square', False)
        self.num_classes = kwargs.get('num_classes', 10)
        self.padding_mode = kwargs.get('padding_mode', 'center')
        self.normalize_size = kwargs.get('normalize_size', False)

    def __len__(self):
        return len(self.task_set) if self.task_set else 1

    def __getitem__(self, idx):

        task = self.task_set[idx]

        # un-Parsing
        grid_max_H = 0
        grid_max_W = 0
        
        grids_raw = []
        for task_subset in [task.train, task.test]:
            for Gin, Gout in task_subset:
                grids_raw += [Gin, Gout]
                max_h = max(Gin.shape[0], Gout.shape[0])
                max_w = max(Gin.shape[1], Gout.shape[1])

                if grid_max_H < max_h:
                    grid_max_H = max_h

                if grid_max_W < max_w:
                    grid_max_W = max_w

        if isinstance(self.grid_size, (list, tuple)):
            grid_size = self.grid_size
        else:
            grids_size = (grid_max_H, grid_max_W)
            if self.grid_square:
                grids_size = tuple([max(*grids_size)] * 2)

        # Padding
        grids_pad = []
        for grid in grids_raw:

            grid_upscale = max(*grids_size) // np.max(grid.shape)
            if (grid_upscale > 2) and self.normalize_size:
                grid = scale_up(grid, h_scale=grid_upscale, 
                                      v_scale=grid_upscale)
            if grid.shape != grids_size:
                grid = self.padding(grid, grids_size=grids_size)

            grids_pad.append(grid)

        if self.batch_size > 0:
            grids_pad = rd.choice(grids_pad, k=self.batch_size)

        grids_pad = torch.tensor(grids_pad)
        return [grids_pad] * 2

    def padding(self, grid, grids_size = None):

        assert isinstance(grid_size, (list, tuple)) \
        or isinstance(self.grid_size, (list, tuple))

        if grid_size is None:
            grid_size = self.grid_size

        pad_h = grids_size[0] - grid.shape[0]
        pad_v = grids_size[1] - grid.shape[1]

        if self.padding_mode == 'center':
            pads = [(pad_h // 2, pad_h - (pad_h // 2)), 
                    (pad_v // 2, pad_v - (pad_v // 2))]
        else:
            pads = [None, None]

        if 'left' in self.padding_mode:
            pads[0] = (0, pad_h)
        elif 'right' in self.padding_mode:
            pads[0] = (pad_h, 0)

        if 'top' in self.padding_mode:
            pads[1] = (0, pad_v)
        elif 'bot' in self.padding_mode:
            pads[1] = (pad_v, 0)

        assert all([p is not None for p in pads])
        grid = np.pad(grid, pad_width=pads, constant_values=-1)
        return grid

    def randomize(self):
        grid_size = self.grid_size
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size]
        return np.random.randint(low=-1, high=10, size=(self.batch_size, *grid_size))

    def shuffle(self):
        rd.shuffle(self.task_set.tasks)


class iARCDatasetAug(iARCDatasetNaive):
    """
    Identity ARC Dataset with Augmentation
    """
    def __init__(
            self, 
            task_set: TaskSet = None, 
            batch_size: int = None, 
            grid_size: Union[None, int, List[int]] = None, 
            augmentors: Union[None, Callable, List, Tuple] = AUGMENTATIONS, 
          **kwargs
        ):

        super(iARCDatasetAug, self).__init__(task_set, batch_size, grid_size, **kwargs)

        # Add augmentors
        self.augmentors = [None]
        if isinstance(augmentors, (List, Tuple)):
            if not augmentors[0]:
                self.augmentors = augmentors
            elif callable(augmentors[0]):
                self.augmentors = augmentors
            elif isinstance(augmentors[0], (List, Tuple)):
                self.augmentors = iter_product(augmentors)

    def __getitem__(self, idx):
        
        # Get padded grids -> (batch_size, batch_max_height, batch_max_width)
        grids_pad = super(iARCDatasetAug, self)[idx][0]
        grids_pad = grids_pad.detach().cpu().numpy()

        # Augmentation
        grids_aug = []
        for grid in grids_pad:
            aug = rd.choice(self.augmentors)
            if isinstance(aug, List):
                for Fx in aug:
                    grid = Fx(grid)
            elif isinstance(grid, Callable):
                grid = aug(grid)
            grids_aug.append(grid)

        grids_aug = torch.tensor(grids_aug)
        return [grids_aug] * 2


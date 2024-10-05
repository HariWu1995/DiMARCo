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
    flip_vertical, flip_horizontal, flip_channel,
    self_identity, rotate_90, rotate_180, rotate_270,
)


# 2-stage Augmentations: 1 -> 4 -> 16
AUGMENTATIONS = [
    (self_identity, flip_vertical, flip_horizontal, flip_channel),
    (self_identity, rotate_90, rotate_180, rotate_270)
]


class ARCDatasetNaive(Dataset):

    def __init__(self, task_set: TaskSet = None, 
                        batch_size: int = None, 
                        grid_size: Union[None, int, List[int]] = None,
                        augmentors: Union[None, Callable, List, Tuple] = AUGMENTATIONS):

        self.task_set = task_set
        self.augmentors = [None]

        if isinstance(augmentors, (List, Tuple)):
            if not augmentors[0]:
                self.augmentors = augmentors
            elif callable(augmentors[0]):
                self.augmentors = augmentors
            elif isinstance(augmentors[0], (List, Tuple)):
                self.augmentors = iter_product(augmentors)

        self.grid_size = grid_size
        self.batch_size = batch_size if batch_size else -1

    def __len__(self):
        return len(self.task_set) if self.task_set else 0

    def __getitem__(self, idx):
        task = self.task_set[idx]
        
        grids_raw = []
        grid_max_H = 0
        grid_max_W = 0

        # un-Parsing
        for task_subset in [task.train, task.test]:
            for Gin, Gout in task_subset:
                grids_raw += [Gin, Gout]
                max_h = max(Gin.shape[0], Gout.shape[0])
                max_w = max(Gin.shape[1], Gout.shape[1])

                if grid_max_H < max_h:
                    grid_max_H = max_h

                if grid_max_W < max_w:
                    grid_max_W = max_w

        # Padding
        grids_pad = []
        grids_size = (grid_max_H, grid_max_W)
        for grid in grids_raw:
            if grid.shape != grids_size:
                pads = [(0, grids_size[0] - grid.shape[0]), 
                        (0, grids_size[1] - grid.shape[1])]
                grid = np.pad(grid, pad_width=pads, constant_values=-1)
            grids_pad.append(grid)

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

        if self.batch_size > 0:
            grids_aug = rd.choice(grids_aug, k=self.batch_size)
        return grids_aug

    def randomize(self, grid_size: Union[int, List[int]] = None):
        grid_size = grid_size if grid_size else self.grid_size
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size]
        return np.random.randint(low=-1, high=10, size=(self.batch_size, *grid_size))
        



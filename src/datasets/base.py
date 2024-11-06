"""
Base-DataLoader
"""
from typing import List, Tuple, Union, Callable

import random as rd
import numpy as np

import torch
from torch.utils.data import Dataset

from src.arckit.data import TaskSet, Task


class ARCBaseDataset(Dataset):

    def __init__(
            self, 
            task_set: Union[TaskSet, Task], 
            grid_size: Union[int, List[int]], 
          **kwargs
        ):

        self.task_set = task_set
        self.grid_size = [grid_size] * 2 if isinstance(grid_size, int) else grid_size
        self.grid_square = kwargs.get('grid_square', False)
        self.num_classes = kwargs.get('num_classes', 10)
        self.padding_mode = kwargs.get('padding_mode', 'center')
        self.padding_value = kwargs.get('padding_value', -1)
        self.normalize_size = kwargs.get('normalize_size', False)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def shuffle(self):
        raise NotImplementedError

    def padding(self, grid, grid_size = None):

        assert isinstance(grid_size, (list, tuple)) \
        or isinstance(self.grid_size, (list, tuple))

        if grid_size is None:
            grid_size = self.grid_size

        pad_h = grid_size[0] - grid.shape[0]
        pad_v = grid_size[1] - grid.shape[1]

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
        grid = np.pad(grid, pad_width=pads, constant_values=self.padding_value)
        return grid

    def randomize(self):
        grid_size = self.grid_size
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size]
        return np.random.randint(low=-1, high=10, size=(self.batch_size, *grid_size))



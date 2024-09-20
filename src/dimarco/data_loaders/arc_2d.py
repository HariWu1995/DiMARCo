"""
DataLoader for naïve ARC 2D-format
"""
import numpy as np

import torch
from torch.utils.data import Dataset

from arckit.data import Task, TaskSet

from dimarco.data_loaders.augmentations import (
    flip_vertical, flip_horizontal, flip_channel,
    rotate_90, rotate_180, rotate_270,
)


# 2-stage Augmentations: 1 -> 4 -> 16
AUGMENTATIONS = [
    (None, flip_vertical, flip_horizontal, flip_channel),
    (None, rotate_90, rotate_180, rotate_270)
]


class ARCDatasetNaive(Dataset):

    def __init__(self, task_set: TaskSet, augmentors=AUGMENTATIONS):
        pass


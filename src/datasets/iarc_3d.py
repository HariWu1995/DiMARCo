"""
DataLoader to enrich data
    from naïve ARC 2D-format (H x W)
    to multi-layer 3D-format (H x W x C), 
                                where C = 9 is number of non-background values (non-black colors)
"""
import numpy as np

import torch
from torch.utils.data import Dataset

from arckit.data import Task, TaskSet


class ARCDatasetDepth(Dataset)



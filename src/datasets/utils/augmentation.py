"""
Sample Augmentation 
"""
import torch
import numpy as np


def self_identity(grid):
    return grid


def flip_vertical(grid):
    return torch.flip(grid, dims=[0])  # Flip along rows


def flip_horizontal(grid):
    return torch.flip(grid, dims=[1])  # Flip along columns


def flip_channel(grid):
    dim = len(grid.size())
    if dim == 3:
        return torch.flip(grid, dims=[2])  # Flip along channels (for 3D)
    elif dim == 2:
        return torch.where(grid == 0, grid, 10 - grid)
    else:
        raise ValueError(f"Only support 2D or 3D-tensor while input has shape {rid.size()}")


def rotate_90(grid):
    return torch.rot90(grid, k=1, dims=[0, 1])


def rotate_180(grid):
    return torch.rot90(grid, k=2, dims=[0, 1])


def rotate_270(grid):
    return torch.rot90(grid, k=3, dims=[0, 1])


def encode_1hot(grid, num_classes: int = 10, 
                ignore_background: bool = False):
    """
    1-hot encoding for 2D-data (2D -> 3D)

    Example:

        Input:
                        0 0 1 1 0 0
                        0 0 1 1 0 0
                        0 0 0 0 0 0
                        0 2 2 2 2 0
                        0 2 2 2 2 0

        Output:
            + C = 0:    1 1 0 0 1 1
                        1 1 0 0 1 1 
                        1 1 1 1 1 1
                        1 0 0 0 0 1
                        1 0 0 0 0 1

            + C = 1:    0 0 1 1 0 0
                        0 0 1 1 0 0
                        0 0 0 0 0 0
                        0 0 0 0 0 0
                        0 0 0 0 0 0

            + C = 2:    0 0 0 0 0 0
                        0 0 0 0 0 0
                        0 0 0 0 0 0
                        0 1 1 1 1 0
                        0 1 1 1 1 0
            
            + others: all-0 arrays
    """
    if num_classes is None:
        num_classes = 10
    if num_classes < 0:
        num_classes = 10

    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()

    from tensorflow.keras.utils import to_categorical
    
    grid_3d = to_categorical(grid, num_classes=num_classes)
    
    if ignore_background:
        grid_3d = grid_3d[..., 1:]

    return torch.tensor(grid_3d)



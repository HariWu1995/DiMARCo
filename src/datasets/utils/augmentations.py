"""
Sample Augmentation 
"""
import torch
import numpy as np


def self_identity(grid):
    return grid


def scale_up(grid, h_scale: int = 2, v_scale: int = 2):

    if isinstance(grid, np.ndarray):
        grid = np.repeat(grid, repeats=h_scale, axis=-2)
        grid = np.repeat(grid, repeats=v_scale, axis=-1)
        return grid

    grid = torch.repeat_interleave(grid, repeats=h_scale, dim=-2)
    grid = torch.repeat_interleave(grid, repeats=v_scale, dim=-1)
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



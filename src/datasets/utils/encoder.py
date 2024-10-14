import numpy as np
import torch

# from tensorflow.keras.utils import to_categorical
    
def to_categorical(x, num_classes = None):

    if num_classes is None:
        num_classes = np.max(x) + 1

    in_shape = x.shape
    out_shape = in_shape + (num_classes,)

    x_min = np.min(x)
    if x_min < 0:
        x -= x_min
        num_classes -= x_min

    x = np.array(x, dtype="int64")
    x = x.reshape(-1)

    num_samples = x.shape[0]
    x_cat = np.zeros((num_samples, num_classes))
    x_cat[np.arange(num_samples), x] = 1

    if x_min < 0:
        x_cat = x_cat[..., -x_min:]

    x_cat = np.reshape(x_cat, out_shape)
    return x_cat


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
        num_classes = torch.max(grid) + 1

    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()

    grid_3d = to_categorical(grid, num_classes = num_classes)
    
    # ARC background-value = 0, not padding-value = -1
    if ignore_background:
        grid_3d = grid_3d[..., 1:]

    return torch.tensor(grid_3d).to(torch.int)


if __name__ == "__main__":

    x = np.random.randint(low=-1, high=10, size=(5, 5), dtype=int)
    x = [[-1, 0, 0, 0],
         [-1, 1, 1, 0],
         [-1, 1, 1, 0],
         [-1, 0, 0, 0]]
    x = torch.tensor(x).unsqueeze(dim=0)
    print(x.shape)
    print(x)

    y = encode_1hot(x)
    print(y.shape)
    print(y)


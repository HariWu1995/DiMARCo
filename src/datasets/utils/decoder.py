import numpy as np


def find_argmax(x, n: int = 1):
    old_shape = x.shape
    new_shape = old_shape[:-n] + (np.prod(old_shape[-n:]),)
    
    max_idx = x.reshape(new_shape).argmax(-1)
    return np.unravel_index(max_idx, new_shape[-n:])[0]


def decode_1hot(grid, add_padding_layer: bool = True, 
                    padding_threshold: float = 0.169):

    if add_padding_layer:
        shape = list(grid.shape)
        shape[1] += 1
        temp = np.ones(shape) * padding_threshold
        temp[:, 1:, ...] = grid
        grid = temp

    grid = np.swapaxes(
           np.swapaxes(grid, 0, 2), 0, 1)
    grid = find_argmax(grid)
    return grid


if __name__ == "__main__":

    from .encoder import encode_1hot

    x = np.random.randint(low=-1, high=10, size=(1, 5, 5), dtype=int)    
    print(x)

    y = encode_1hot(x)
    z = decode_1hot(y)
    print(z)


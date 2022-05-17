from typing import Tuple

import numpy as np


def get_indices(x_shape: Tuple[int, int, int, int],
                filter_height: int,
                filter_width: int,
                stride: int,
                padding: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size, n_channels, height, width = x_shape

    out_h = (height + 2 * padding - filter_height) // stride + 1
    out_w = (width + 2 * padding - filter_width) // stride + 1

    level1 = np.repeat(np.arange(filter_height), filter_width)
    level1 = np.tile(level1, n_channels)
    every_levels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + every_levels.reshape(1, -1)

    slide1 = np.tile(np.arange(filter_width), filter_height)
    slide1 = np.tile(slide1, n_channels)
    every_slides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + every_slides.reshape(1, -1)

    d = np.repeat(np.arange(n_channels), filter_height * filter_width).reshape(-1, 1)

    return i, j, d


def im2col(x: np.ndarray,
           filter_height: int,
           filter_width: int,
           stride: int,
           padding: int) -> np.ndarray:
    x_padded = np.pad(x,
                      ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                      mode='constant')
    i, j, d = get_indices(x.shape, filter_height, filter_width, stride, padding)
    col = x_padded[:, d, i, j]
    col = np.concatenate(col, axis=-1)
    return col


def col2im(dx_col: np.ndarray,
           x_shape: Tuple[int, int, int, int],
           filter_height: int,
           filter_width: int,
           stride: int,
           padding: int) -> np.ndarray:
    n, d, h, w = x_shape
    h_padded, w_padded = h + 2 * padding, w + 2 * padding
    x_padded = np.zeros((n, d, h_padded, w_padded))

    i, j, d = get_indices(x_shape, filter_height, filter_width, stride, padding)
    dx_col_reshaped = np.array(np.hsplit(dx_col, n))
    np.add.at(x_padded, (slice(None), d, i, j), dx_col_reshaped)
    if padding == 0:
        return x_padded
    else:
        return x_padded[:, :, padding:-padding, padding:-padding]

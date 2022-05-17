import numpy as np

from ..base import Module
from .functional import im2col, col2im


class MeanPool(Module):
    def __init__(self,
                 filter_size: int,
                 stride: int = 1,
                 padding: int = 0) -> None:
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.x_shape = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape

        batch_size, n_channels, in_height, in_width = x.shape
        out_height = (in_height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.filter_size) // self.stride + 1

        x_col = im2col(x, self.filter_size, self.filter_size, self.stride, self.padding)
        x_col = x_col.reshape(n_channels, x_col.shape[0] // n_channels, -1)
        a_pool = np.mean(x_col, axis=1)
        a_pool = np.array(np.hsplit(a_pool, batch_size))
        a_pool = a_pool.reshape(batch_size, n_channels, out_height, out_width)

        return a_pool

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size, n_channels, in_height, in_width = self.x_shape

        dout_flatten = dout.reshape(n_channels, -1) / (self.filter_size * self.filter_size)
        dx_col = np.repeat(dout_flatten, self.filter_size * self.filter_size, axis=0)
        dx = col2im(dx_col, self.x_shape, self.filter_size, self.filter_size, self.stride, self.padding)
        dx = dx.reshape(batch_size, -1)
        dx = np.array(np.hsplit(dx, n_channels))
        dx = dx.reshape(batch_size, n_channels, in_height, in_width)
        return dx

from typing import List

import numpy as np

from ..base import Module, Parameter
from .functional import im2col, col2im


class Conv2D(Module):
    """
    ref: https://github.com/3outeille/CNNumpy/blob/master/src/fast/layers.py
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filter_size: int,
                 stride: int = 1,
                 padding: int = 0) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.W_shape = (self.out_channels, self.in_channels, self.filter_size, self.filter_size)
        self.W = np.random.randn(*self.W_shape) * np.sqrt(1. / self.filter_size)
        self.dW = None
        self.b = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)
        self.db = None

        self.x_shape = None
        self.x_col = None
        self.W_col = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, in_channels, in_width, in_height = self.x_shape = x.shape

        out_height = (in_height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.filter_size) // self.stride + 1

        self.x_col = im2col(x, self.filter_size, self.filter_size, self.stride, self.padding)
        self.W_col = self.W.reshape((self.out_channels, -1))
        b_col = self.b.reshape((-1, 1))

        out = self.W_col @ self.x_col + b_col

        shape = (batch_size, self.out_channels, out_height, out_width)
        out = np.array(np.hsplit(out, batch_size)).reshape(shape)

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size = self.x_shape[0]
        self.db = np.sum(dout, axis=(0, 2, 3))

        d0, d1, d2, d3 = dout.shape
        dout = dout.reshape(d0 * d1, d2 * d3)
        dout = np.array(np.vsplit(dout, batch_size))
        dout = np.concatenate(dout, axis=-1)
        dx_col = self.W_col.T @ dout
        dW_col = dout @ self.x_col.T
        dx = col2im(dx_col, self.x_shape, self.filter_size, self.filter_size, self.stride, self.padding)

        self.dW = dW_col.reshape(self.W_shape)
        return dx

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(self, 'W'),
            Parameter(self, 'b'),
        ]

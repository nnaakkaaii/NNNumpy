import numpy as np

from ..base import Module


class Dropout(Module):
    def __init__(self, dropout_ratio: float) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask

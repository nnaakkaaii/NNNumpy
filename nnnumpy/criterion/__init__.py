import numpy as np

from ..base import Loss


class BCEWithLogitsLoss(Loss):
    def __init__(self) -> None:
        self.log_y = None
        self.t = None

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        x_max = x.max(axis=1, keepdims=True)
        self.log_y = x - x_max - np.log(np.sum(np.exp(x - x_max), axis=1, keepdims=True))
        self.t = t
        return - np.mean(np.sum(t * self.log_y))

    def backward(self) -> np.ndarray:
        batch_size = self.t.shape[0]
        y = np.exp(self.log_y)
        return (y - self.t) / batch_size

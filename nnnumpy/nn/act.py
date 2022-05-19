import numpy as np

from ..base import Module


class Sigmoid(Module):
    def __init__(self) -> None:
        self.y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.clip(min=-700)  # overflow対策
        y = 1/(1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.y * (1 - self.y)  # 逆伝播計算


class ReLU(Module):
    def __init__(self) -> None:
        self.x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x * (x > 0)  # 順伝播計算

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return np.where(self.x > 0, dout, 1e-7)


class Tanh(Module):
    def __init__(self) -> None:
        self.y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.y = np.tanh(x)
        return self.y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (1 - self.y ** 2)

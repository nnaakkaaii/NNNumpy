from typing import List

import numpy as np

from ..base import Module, Parameter


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None
        self.x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # 順伝播計算
        self.x = x
        u = x @ self.W + self.b
        return u

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self.x.T @ dout
        self.db = np.ones(len(self.x)) @ dout
        return dout @ self.W.T

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(self, 'W'),
            Parameter(self, 'b'),
        ]

from typing import List

import numpy as np

from ..base import Module, Parameter


class Reshape(Module):
    def __init__(self, shape) -> None:
        self.before = None
        self.after = shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.before is None:
            self.before = x.shape[1:]
        return x.reshape(x.shape[0:1] + self.after)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(dout.shape[0:1] + self.before)


class ModuleList(Module):
    def __init__(self, module_list: List[Module]) -> None:
        self.layers = module_list

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            # print(layer.__class__.__name__)
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> List[Parameter]:
        ret = []
        for layer in self.layers:
            ret += layer.parameters()
        return ret

    def train(self) -> None:
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        super().eval()
        for layer in self.layers:
            layer.eval()

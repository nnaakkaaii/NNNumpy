import abc
import dataclasses
from typing import List

import numpy as np


class Module(metaclass=abc.ABCMeta):
    training: bool

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass

    def parameters(self) -> List['Parameter']:
        return []

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


@dataclasses.dataclass(frozen=False)
class Parameter:
    module: Module
    key: str

    @property
    def param(self) -> np.ndarray:
        assert hasattr(self.module, self.key)
        return getattr(self.module, self.key)

    @param.setter
    def param(self, param: np.ndarray) -> None:
        setattr(self.module, self.key, param)

    @property
    def grad(self) -> np.ndarray:
        assert hasattr(self.module, f'd{self.key}')
        return getattr(self.module, f'd{self.key}')


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self) -> np.ndarray:
        pass


class Optimizer(metaclass=abc.ABCMeta):
    lr: float

    @abc.abstractmethod
    def step(self) -> None:
        pass

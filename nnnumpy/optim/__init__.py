from typing import List

import numpy as np

from ..base import Optimizer, Parameter


class SGDOptimizer(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            p.param -= self.lr * p.grad


class AdamOptimizer(Optimizer):
    def __init__(self, parameters: List[Parameter], lr, beta1=0.9, beta2=0.999):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1  # mの減衰率
        self.beta2 = beta2  # vの減衰率
        self.iter = 0  # 試行回数を初期化
        self.m = [np.zeros_like(p.param) for p in parameters]  # モーメンタム
        self.v = [np.zeros_like(p.param) for p in parameters]  # 適合的な学習係数

    def step(self):
        self.iter += 1  # 更新回数をカウント
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)  # 学習率
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            p.param -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

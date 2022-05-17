from typing import Optional

import numpy as np

from ..base import Module


class BatchNormalization(Module):
    def __init__(self,
                 gamma: float,
                 beta: float,
                 momentum: float = 0.9,
                 running_mean: Optional[np.ndarray] = None,
                 running_var: Optional[np.ndarray] = None):
        """
        :param gamma: 標準偏差
        :param beta: 平均
        :param momentum: 減衰率
        :param running_mean: テスト時に用いる平均値
        :param running_var: テスト時に用いる分散
        """

        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None
        self.dgamma = None
        self.dbeta = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if self.training:
            self.batch_size = x.shape[0]
            mu = x.mean(axis=0)  # 平均
            self.xc = x - mu  # 偏差
            var = np.mean(self.xc ** 2, axis=0)  # 分散
            self.std = np.sqrt(var + 10e-7)  # 標準偏差
            xn = self.xc / self.std  # 標準化
            self.xn = xn
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu  # 過去の平均の情報
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var  # 過去の分散の情報
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 10e-7)

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dbeta = dout.sum(axis=0)  # 調整後の平均
        self.dgamma = np.sum(self.xn * dout, axis=0)  # 調整後の標準偏差
        dxn = self.gamma * dout  # 正規化後のデータ
        dxc = dxn / self.std  # 偏差
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)  # 標準偏差
        dvar = 0.5 * dstd / self.std  # 分散
        dxc += (2.0 / self.batch_size) * self.xc * dvar  # 偏差
        dmu = np.sum(dxc, axis=0)  # 平均
        dx = dxc - dmu / self.batch_size  # 入力データ

        return dx
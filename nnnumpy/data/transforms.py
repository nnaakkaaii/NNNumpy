from typing import Tuple

import cv2
import numpy as np

from ..base import Transform


class NoTransform(Transform):
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class RotateTransform(Transform):
    def __init__(self,
                 max_angle: float,
                 max_scale: float,
                 dsize: Tuple[int, int]) -> None:
        self.max_angle = max_angle
        self.max_scale = max_scale
        self.dsize = dsize
        self.center = (dsize[0]/2, dsize[1]/2)

    def transform(self, x: np.ndarray) -> np.ndarray:
        # max_angle=15なら、-15~15
        angle = 2. * self.max_angle * np.random.rand() - self.max_angle
        # max_scale=0.2なら、0.8~1.2
        scale = 1. + 2. * self.max_scale * np.random.rand() - self.max_scale
        mat = cv2.getRotationMatrix2D(self.center, angle=angle, scale=scale)
        ret = cv2.warpAffine(src=x.reshape(self.dsize), M=mat, dsize=self.dsize)
        return ret.reshape(-1)

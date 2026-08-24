import numpy as np

from typing import List


class Solution:

    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x_np = np.array(x, dtype=float)
        gamma_np = np.array(gamma, dtype=float)

        rms = np.sqrt(np.mean(x_np ** 2) + eps)
        x_norm = x_np / rms

        y = gamma_np * x_norm

        return np.round(y, 4).tolist()
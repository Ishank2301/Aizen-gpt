import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
        n = len(y_true)

        # Keep probabilities strictly between 0 and 1
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        ) / n

        return round(float(loss), 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
        n = len(y_true)

        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / n

        return round(float(loss), 4)

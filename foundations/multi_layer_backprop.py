import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Convert inputs to NumPy arrays
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        # Forward pass
        # W1 shape: (hidden_size, input_size)
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)

        # W2 shape: (output_size, hidden_size)
        z2 = W2 @ a1 + b2
        y_pred = z2

        # MSE loss
        n = y_true.size
        error = y_pred - y_true
        loss = np.mean(error ** 2)

        # Backward pass

        # dL/dz2
        dz2 = (2.0 / n) * error

        # Gradients for second linear layer
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        # Gradient flowing into a1
        da1 = W2.T @ dz2

        # ReLU derivative: 1 when z1 > 0, otherwise 0
        dz1 = da1 * (z1 > 0)

        # Gradients for first linear layer
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }

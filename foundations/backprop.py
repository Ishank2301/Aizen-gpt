import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
# Step 1: Forward Pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        
        # Step 2: Gradients using the Chain Rule
        # 2a. Derivative of Loss with respect to y_hat (dL/dy_hat)
        # L = 0.5 * (y_hat - y_true)^2 -> dL/dy_hat = (y_hat - y_true)
        dL_dyhat = y_hat - y_true
        
        # 2b. Derivative of y_hat with respect to z (dy_hat/dz)
        # Derivative of Sigmoid: σ(z) * (1 - σ(z))
        dyhat_dz = y_hat * (1 - y_hat)
        
        # 2c. Combine to find Derivative of Loss with respect to z (dL/dz)
        dL_dz = dL_dyhat * dyhat_dz
        
        # Step 3: Compute final gradients for weights and bias
        # dL/dw = dL/dz * dz/dw (where dz/dw is just x)
        dL_dw = dL_dz * x
        
        # dL/db = dL/dz * dz/db (where dz/db is just 1)
        dL_db = dL_dz * 1.0
        
        # Step 4: Round and return
        return np.round(dL_dw, 5), round(float(dL_db), 5)

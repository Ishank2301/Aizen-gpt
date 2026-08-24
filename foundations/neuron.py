import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        # Step 1: Calculate the pre-activation (z = x·w + b)
        z = np.dot(x, w) + b
        
        # Step 2: Apply the requested activation function
        if activation == "sigmoid":
            my_ans = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            my_ans = max(0.0, float(z)) # np.maximum(0, z) works too
        else:
            raise ValueError("Activation must be 'sigmoid' or 'relu'")
            
        # Step 3: Round and return as a standard Python float
        return round(float(my_ans), 5)

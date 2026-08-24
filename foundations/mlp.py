import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        
        current_activation = x
        num_layers = len(weights)
        
        for i in range(num_layers):
            w = weights[i]
            b = biases[i]
            
            # 1. Bulletproof Matrix Math (Handles both shape conventions)
            if w.shape[0] == current_activation.shape[0]:
                # Standard convention: W is (in_features, out_features)
                z = np.dot(current_activation, w) + b
            else:
                # PyTorch convention: W is (out_features, in_features)
                z = np.dot(w, current_activation) + b
                
            # 2. Activation Logic
            # Apply ReLU to all hidden layers, but skip it for the final output layer
            if i < num_layers - 1:
                current_activation = np.maximum(0.0, z)
            else:
                current_activation = z
                
        # 3. Return final predictions rounded to 5 decimal places
        return np.round(current_activation, 5)
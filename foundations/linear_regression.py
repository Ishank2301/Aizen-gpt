import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Matrix multiplication using np.dot or the @ operator
        predictions = np.dot(X, weights)
        
        # Round the resulting array to 5 decimal places
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute Mean Squared Error (MSE)
        # Square the differences first, then find the mean
        mse = np.mean((ground_truth - model_prediction) ** 2)
        
        # Round the final scalar float to 5 decimal places
        return round(float(mse), 5)
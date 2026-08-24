import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        
        # 0. Make a copy of the weights so we don't mutate the original input array
            weights = initial_weights.copy()
            N = len(X)
            num_weights = len(weights)
        
            for _ in range(num_iterations):
            # 1. Compute predictions for the current iteration
                predictions = self.get_model_prediction(X, weights)
            
            # Create an array to temporarily hold gradients for this step
                gradients = np.zeros(num_weights)
            
            # 2. For each weight index j, compute gradient with get_derivative()
                for j in range(num_weights):
                    gradients[j] = self.get_derivative(predictions, Y, N, X, j)
                
            # 3. Update all weights simultaneously: weights[j] -= learning_rate * gradient
                for j in range(num_weights):
                    weights[j] -= self.learning_rate * gradients[j]
                
        # Return the final weights rounded to 5 decimal places
            return np.round(weights, 5)

import numpy as np
from typing import Tuple, List

# internally batch normalization is z score norm
class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
                   
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            mu = np.mean(x, axis = 0)
            var = np.var(x, axis = 0)
            x_hat = (x - mu) / np.sqrt(var + eps)  # z-score
            running_mean = (1 - momentum) * running_mean + momentum * mu
            running_var = (1 - momentum) * running_var + momentum * var
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        
        y_pred = gamma * x_hat + beta

        return (np.round(y_pred, 4), np.round(running_mean, 4), np.round(running_var, 4))

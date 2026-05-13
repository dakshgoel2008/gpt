import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        
        x = np.array(x, dtype = float)
        gamma = np.array(gamma, dtype = float)
        rms = np.sqrt(np.mean(x**2 + eps))

        x_hat = x / rms
        y_pred = gamma * x_hat

        return np.round(y_pred, 4)
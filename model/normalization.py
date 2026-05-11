import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D -> feature vector
        # gamma: 1D (same length as x)
        # beta: 1D (same length as x)
        
        eps = 1e-5
        x_hat = (x - np.mean(x)) / math.sqrt(np.var(x) + eps)
        output = gamma * x_hat + beta
        return np.round(output, 5)

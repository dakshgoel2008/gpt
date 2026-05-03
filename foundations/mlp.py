import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        
        layers = len(weights) - 1  # no. of hidden layers excluding the output layer
        for i in range(layers):
            x = np.maximum(0, x @ weights[i] + biases[i])

        x = x @ weights[-1] + biases[-1]
        return np.round(x, 5)

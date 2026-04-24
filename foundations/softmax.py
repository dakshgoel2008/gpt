import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        return np.round(
            np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z))), 
            4
        )
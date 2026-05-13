import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n, m = X.shape
        w = np.zeros(m)
        b = 0
        for _ in range(epochs):
            y_hat = X @ w + b
            MSE = np.mean((y_hat - y)**2)
            dL_dw = (2 / n) * (X.T @ (y_hat - y))
            dL_db = (2 / n) * np.sum(y_hat - y)

            w = w - lr * dL_dw
            b = b - lr * dL_db
        return (np.round(w, 5), np.round(b, 5))

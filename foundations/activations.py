import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        out = np.empty_like(z)
        np.negative(z, out=out)
        np.exp(out, out=out)
        out += 1
        np.reciprocal(out, out=out)
        np.around(out, 5, out=out)
        return out

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        out = np.empty_like(z)
        np.maximum(0, z, out=out)
        return out

import numpy as np
from numpy.typing import NDArray


class Solution:
    def __init__(self) -> None:
        # cached buffer
        self._buf_sigmoid: NDArray[np.float64] | None = None
        self._buf_relu: NDArray[np.float64] | None = None

    def _get_buf(self, buf: NDArray | None, z: NDArray[np.float64]) -> NDArray[np.float64]:
        if buf is None or buf.shape != z.shape:
            return np.empty_like(z)
        return buf

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        self._buf_sigmoid = self._get_buf(self._buf_sigmoid, z)
        np.negative(z, out=self._buf_sigmoid)
        np.exp(self._buf_sigmoid, out=self._buf_sigmoid)
        self._buf_sigmoid += 1
        np.reciprocal(self._buf_sigmoid, out=self._buf_sigmoid)
        return np.round(self._buf_sigmoid, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        self._buf_relu = self._get_buf(self._buf_relu, z)
        np.maximum(0, z, out=self._buf_relu)
        return self._buf_relu
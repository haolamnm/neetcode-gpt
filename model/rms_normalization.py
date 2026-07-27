import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_np = np.asarray(x, dtype=np.float64)
        g_np = np.asarray(gamma, dtype=np.float64)

        rms = np.sqrt(np.mean(x_np ** 2) + eps)
        x_hat = x_np / rms
        output = g_np * x_hat

        return np.round(output, 4)

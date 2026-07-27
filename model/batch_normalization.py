import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x_np = np.asarray(x, dtype=np.float64)
        gamma_np = np.asarray(gamma, dtype=np.float64)
        beta_np = np.asarray(beta, dtype=np.float64)
        
        running_mean_np = np.asarray(running_mean, dtype=np.float64)
        running_var_np = np.asarray(running_var, dtype=np.float64)

        nuy_b = np.mean(x_np, axis=0)
        var_b = np.var(x_np, axis=0)

        if training:
            running_mean_np = (1 - momentum) * running_mean_np + momentum * nuy_b
            running_var_np = (1 - momentum) * running_var_np + momentum * var_b
            x_hat = (x_np - nuy_b) / np.sqrt(var_b + eps)

        else:
            x_hat = (x_np - running_mean_np) / np.sqrt(running_var_np + eps)

        y_np = gamma_np * x_hat + beta_np

        return (
            np.round(y_np, 4), 
            np.round(running_mean_np, 4), 
            np.round(running_var_np, 4)
        )
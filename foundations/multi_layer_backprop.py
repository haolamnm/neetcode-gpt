import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x_arr = np.asarray(x, dtype=np.float64)
        W1_arr = np.asarray(W1, dtype=np.float64)
        b1_arr = np.asarray(b1, dtype=np.float64)
        W2_arr = np.asarray(W2, dtype=np.float64)
        b2_arr = np.asarray(b2, dtype=np.float64)
        y_true_arr = np.asarray(y_true, dtype=np.float64)
        
        # 1. FORWARD PASS
        z1 = W1_arr @ x_arr + b1_arr
        a1 = np.maximum(0, z1)
        z2 = W2_arr @ a1 + b2_arr
        y_hat = z2
        
        # Loss: Mean Squared Error
        loss = np.mean((y_hat - y_true_arr) ** 2)

        # 2. BACKWARD PASS
        N = len(y_true_arr)
        
        dL_dz2 = (2.0 / N) * (y_hat - y_true_arr)
        
        # Layer 2 gradients
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2
        
        # Backprop through Layer 2 to hidden layer: dL/da1 = W2^T @ dL/dz2
        dL_da1 = W2_arr.T @ dL_dz2
        
        # Backprop through ReLU: dL/dz1 = dL/da1 * (z1 > 0)
        dL_dz1 = dL_da1.copy()
        dL_dz1[z1 <= 0] = 0.0
        
        # Layer 1 gradients
        dL_dW1 = np.outer(dL_dz1, x_arr)
        dL_db1 = dL_dz1

        return {
            "loss": float(np.round(loss, 4)),
            "dW1": np.round(dL_dW1, 4).tolist(),
            "db1": np.round(dL_db1, 4).tolist(),
            "dW2": np.round(dL_dW2, 4).tolist(),
            "db2": np.round(dL_db2, 4).tolist()
        }
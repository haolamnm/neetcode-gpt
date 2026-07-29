import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fracs = []

        with torch.no_grad():
            curr_x = x
            modules = (
                list(model.children()) if list(model.children()) else [model]
            )

            for module in modules:
                curr_x = module(curr_x)

                if isinstance(module, nn.ReLU):
                    dead_mask = (curr_x == 0).all(dim=0)
                    dead_frac = round(dead_mask.float().mean().item(), 4)
                    dead_fracs.append(dead_frac)

        return dead_fracs

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if not dead_fractions:
            return "healthy"

        # Rule 1: 'use_leaky_relu' if any layer has dead fraction > 0.5
        if any(f > 0.5 for f in dead_fractions):
            return "use_leaky_relu"

        # Rule 2: 'reinitialize' if the first layer has dead fraction > 0.3
        if dead_fractions[0] > 0.3:
            return "reinitialize"

        # Rule 3: 'reduce_learning_rate' if dead fractions strictly increase with depth
        # AND the last layer's fraction > 0.1
        is_strictly_increasing = len(dead_fractions) > 1 and all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )

        if is_strictly_increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

        # Rule 4: 'healthy' if max dead fraction < 0.1
        if max(dead_fractions) < 0.1:
            return "healthy"

        # Rule 5: 'healthy' otherwise
        return "healthy"

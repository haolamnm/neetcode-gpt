import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():
            curr_x = x
            for module in model.children() if list(model.children()) else [model]:
                curr_x = module(curr_x)

                if isinstance(module, nn.Linear):
                    # curr_x shape: (batch_size, num_neurons)
                    mean_val = round(curr_x.mean().item(), 4)
                    std_val = round(curr_x.std(unbiased=True).item(), 4)

                    # A neuron is dead if it outputs <= 0 for ALL samples in the batch
                    # (curr_x <= 0).all(dim=0) checks each neuron across batch dim 0
                    dead_mask = (curr_x <= 0).all(dim=0)
                    dead_frac = round(dead_mask.float().mean().item(), 4)

                    stats.append(
                        {
                            "mean": mean_val,
                            "std": std_val,
                            "dead_fraction": dead_frac,
                        }
                    )

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        loss_fn = nn.MSELoss()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()

        stats = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                if grad is not None:
                    mean_val = round(grad.mean().item(), 4)
                    std_val = round(grad.std(unbiased=True).item(), 4)
                    norm_val = round(torch.norm(grad, p=2).item(), 4)

                    stats.append(
                        {"mean": mean_val, "std": std_val, "norm": norm_val}
                    )

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # Priority 1: Dead neurons
        if any(stat["dead_fraction"] > 0.5 for stat in activation_stats):
            return "dead_neurons"

        # Priority 2: Exploding gradients via weight gradient norm (> 1000)
        if any(stat["norm"] > 1000 for stat in gradient_stats):
            return "exploding_gradients"

        # Priority 3: Vanishing gradients via last layer's weight gradient norm (< 1e-5)
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # Priority 4: Vanishing gradients via activation std (< 0.1)
        if any(stat["std"] < 0.1 for stat in activation_stats):
            return "vanishing_gradients"

        # Priority 5: Exploding gradients via activation std (> 10.0)
        if any(stat["std"] > 10.0 for stat in activation_stats):
            return "exploding_gradients"

        # Priority 6: Healthy network
        return "healthy"

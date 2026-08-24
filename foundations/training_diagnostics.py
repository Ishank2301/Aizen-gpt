import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> List[Dict[str, float]]:

        stats = []

        with torch.no_grad():
            output = x

            for layer in model.children():

                output = layer(output)

                if isinstance(layer, nn.Linear):
                    stats.append({
                        "mean": round(output.mean().item(), 4),
                        "std": round(output.std().item(), 4),
                        "dead_fraction": 0.0
                    })

                elif isinstance(layer, nn.ReLU):
                    dead_neurons = (output == 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean().item()

                    if stats:
                        stats[-1]["dead_fraction"] = round(
                            dead_fraction, 4
                        )

        return stats

    def compute_gradient_stats(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> List[Dict[str, float]]:

        model.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]]
    ) -> str:

        if any(
            stat["dead_fraction"] > 0.5
            for stat in activation_stats
        ):
            return "dead_neurons"

        if any(
            stat["norm"] > 10.0
            for stat in gradient_stats
        ):
            return "exploding_gradients"

        if any(
            stat["norm"] < 1e-6
            for stat in gradient_stats
        ):
            return "vanishing_gradients"

        return "healthy"
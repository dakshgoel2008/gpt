import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:

        stats = []

        with torch.no_grad():

            for layer in model.children():

                x = layer(x)

                if isinstance(layer, nn.Linear):

                    mean_val = round(x.mean().item(), 4)

                    std_val = round(
                        x.std().item(), 4
                    )

                    # dead neuron = neuron inactive for whole batch
                    if x.dim() >= 2:
                        dead_fraction = round(
                            ((x <= 0).all(dim=0))
                            .float()
                            .mean()
                            .item(),
                            4
                        )
                    else:
                        dead_fraction = round(
                            (x <= 0).float().mean().item(), 4
                        )

                    stats.append({
                        "mean": mean_val,
                        "std": std_val,
                        "dead_fraction": dead_fraction
                    })

        return stats

    def compute_gradient_stats(self,model: nn.Module,x: torch.Tensor,y: torch.Tensor) -> List[Dict[str, float]]:

        model.zero_grad()

        pred = model(x)

        loss = nn.MSELoss()(pred, y)

        loss.backward()

        stats = []

        for layer in model.children():

            if isinstance(layer, nn.Linear):

                grad = layer.weight.grad

                if grad is None:
                    continue

                stats.append({
                    "mean": round(grad.mean().item(),4),
                    "std": round(grad.std().item(),4),
                    "norm": round(grad.norm().item(),4)
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]]
    ) -> str:

        # 1. Dead neurons
        for s in activation_stats:
            if s["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for s in gradient_stats:
            if s["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradients
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation instability
        for s in activation_stats:

            if s["std"] < 0.1:
                return "vanishing_gradients"

            if s["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
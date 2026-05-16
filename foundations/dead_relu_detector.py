import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        for layer in model.children():
            x = layer(x)    # forward pass
            # only execute if layer is a ReLU layer
            if isinstance(layer, nn.ReLU):
                dead = (x == 0).all(dim = 0)   # (batch_size, neurons) => so col will give the neurons
                frac = dead.float().mean().item()   # fraction of dead neurons in a particular layer.
                dead_fractions.append(round(frac, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if any(x > 0.5 for x in dead_fractions):
            return "use_leaky_relu"
        
        if dead_fractions[0] > 0.3:
            return "reinitialize"
        
        increasing = all(dead_fractions[i] < dead_fractions[i + 1] for i in range(len(dead_fractions) - 1))

        if increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"
        
        return "healthy"   # if max dead_fractions < 0.1 then also healthy
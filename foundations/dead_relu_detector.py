import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        with torch.no_grad():
            output = x
            for layer in model:
                output = layer(output)
                if isinstance(layer, nn.ReLU):
                    dead_neurons = torch.all(output == 0, dim=0)
                    dead_fraction = torch.mean(dead_neurons.float()).item()
                    dead_fractions.append(round(dead_fraction, 4))
        return dead_fractions
        pass

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(fraction > 0.5 for fraction in dead_fractions):
            return "use_leaky_relu"
        if len(dead_fractions) > 0 and dead_fractions[0] > 0.3:
            return "reinitialize"
        increasing = True
        for i in range(len(dead_fractions) - 1):
            if dead_fractions[i] >= dead_fractions[i + 1]:
                increasing = False
                break
        if increasing and len(dead_fractions) > 0 and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"
        if len(dead_fractions) > 0 and max(dead_fractions) < 0.1:
            return "healthy"
        return "healthy"        
        pass

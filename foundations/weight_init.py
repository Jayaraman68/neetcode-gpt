import torch
import torch.nn as nn
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        return W.tolist()
        pass

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        W = torch.randn(fan_out, fan_in) * std     
        return W.tolist()
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        
        # 1. Define network dimensions across all layers
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        # 2. Generate all layer weights upfront to match the random state sequence
        for i in range(num_layers):
            if init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            elif init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            else:
                std = 1.0  # Naive standard normal strategy
                
            w = torch.randn(dims[i+1], dims[i]) * std
            weights.append(w)
            
        # 3. Generate the input vector after weights have advanced the random state
        x = torch.randn(1, input_dim)
        stds = []
        
        # 4. Process forward propagation sequentially
        for w in weights:
            x = torch.matmul(x, w.t())
            
            if init_type == 'kaiming':
                x = torch.relu(x)
            elif init_type == 'xavier':
                x = torch.tanh(x)
            else:
                x= torch.relu(x)
                    
            stds.append(round(x.std().item(), 2))
            
        return stds
        
        pass

    


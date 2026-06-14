import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        final_loss = None
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            starts = torch.randint(0,len(data) - context_length,(batch_size,))
            X = []
            Y = []
            for i in starts:
                i = i.item()
                X.append(data[i:i + context_length])
                Y.append(data[i + 1:i + context_length + 1])
            X = torch.stack(X)
            Y = torch.stack(Y)
            logits = model(X)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = Y.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        return round(final_loss, 4)
        pass

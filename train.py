
import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float
    ) -> float:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr
        )

        final_loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            starts = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            X = torch.stack([
                data[i:i + context_length]
                for i in starts
            ])

            Y = torch.stack([
                data[i + 1:i + context_length + 1]
                for i in starts
            ])

            optimizer.zero_grad()

            logits = model(X)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                Y.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        return round(final_loss, 4)
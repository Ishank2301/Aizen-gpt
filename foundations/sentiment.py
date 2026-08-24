import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Embedding output: (B, T, 16)
        embeddings = self.embedding(x)

        # Average across sequence length: (B, 16)
        embeddings = embeddings.mean(dim=1)

        # Linear: (B, 1)
        output = self.linear(embeddings)

        # Sigmoid
        output = self.sigmoid(output)

        # Round to 4 decimal places
        return torch.round(output, decimals=4)
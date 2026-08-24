
import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # Key, Query, Value projections
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Project input
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        # Causal mask
        seq_len = embedded.size(1)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=embedded.device)
        )

        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        scores = torch.softmax(scores, dim=2)

        # Weighted values
        output = scores @ V

        return torch.round(output, decimals=4)
import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # Move heads before sequence dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat each KV head for the corresponding query heads
        repeat_factor = self.num_heads // self.num_kv_heads

        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention
        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention = torch.softmax(scores, dim=-1)

        # Attention output
        out = attention @ v

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.num_heads * self.head_dim)

        # Output projection
        out = self.output_proj(out)

        return torch.round(out, decimals=4)
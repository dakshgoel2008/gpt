import torch
import torch.nn as nn
from torchtyping import TensorType
from .attention import SingleHeadAttention

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        head_size = attention_dim // num_heads

        # Create all attention heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, head_size)
            for _ in range(num_heads)
        ])

        # Output projection W_O
        self.output_projection = nn.Linear(
            attention_dim,
            attention_dim,
            bias=False
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run every head
        head_outputs = [head(embedded) for head in self.heads]

        # Concatenate along embedding dimension
        concatenated = torch.cat(head_outputs, dim=2)

        # Output projection
        output = self.output_projection(concatenated)
        return output
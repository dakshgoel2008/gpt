import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        head_size = attention_dim // num_heads

        # Create all attention heads
        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, head_size)
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

        return torch.round(output * 10000) / 10000

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)

            self.key_gen = nn.Linear(
                embedding_dim, attention_dim, bias=False
            )
            self.query_gen = nn.Linear(
                embedding_dim, attention_dim, bias=False
            )
            self.value_gen = nn.Linear(
                embedding_dim, attention_dim, bias=False
            )

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:

            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ k.transpose(1, 2)

            context_length = k.shape[1]
            attention_dim = k.shape[2]

            scores = scores / (attention_dim ** 0.5)

            # Causal mask
            mask = torch.tril(
                torch.ones(context_length, context_length)
            ) == 0

            scores = scores.masked_fill(mask, float('-inf'))

            attention_weights = nn.functional.softmax(scores, dim=2)

            return attention_weights @ v
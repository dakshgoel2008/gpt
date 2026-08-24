import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)

        score = q @ torch.transpose(k, 1, 2)
        context_len, attention_dim = k.shape[1], k.shape[2]
        score = score / (attention_dim**0.5)

        lower_triangular = torch.tril(torch.ones(context_len, context_len))
        mask = lower_triangular == 0
        score = score.masked_fill(mask, float('-inf'))
        score = nn.functional.softmax(score, dim=2)

        return torch.round(score @ v, decimals=4)

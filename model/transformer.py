import torch
import torch.nn as nn
from torchtyping import TensorType
from .multi_head_attention import MultiHeadedSelfAttention

class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        # 1. Multi-Head Self Attention
        self.attention = MultiHeadedSelfAttention(
            embedding_dim=model_dim, attention_dim=model_dim, num_heads=num_heads
        )

        # 2. Feed Forward Neural Network
        self.feed_forward = VanillaNeuralNetwork(model_dim)

        # 3. Two LayerNorm instances
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # First Pre-LN sub-layer + residual
        embedded = embedded + self.attention(self.layer_norm_1(embedded))

        # Second Pre-LN sub-layer + residual
        embedded = embedded + self.feed_forward(self.layer_norm_2(embedded))

        return embedded
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Embedding layer: vocabulary_size in, 16 out
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16)

        # Linear Layer: 16 in, 1 out
        self.linear = nn.Linear(in_features=16, out_features=1)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # x is of shape (B, T) where B is batch_size and T is sequence_length
        
        # Output shape: (B, T, 16)
        embeds = self.embedding(x)
        
        # Average across the sequence length dimension (dim=1)
        # Output shape: (B, 16)
        averaged_embeds = torch.mean(embeds, dim=1)
        
        # Output shape: (B, 1)
        linear_out = self.linear(averaged_embeds)
        
        # Output shape: (B, 1)
        probs = self.sigmoid(linear_out)
        
        return torch.round(probs, decimals=4)

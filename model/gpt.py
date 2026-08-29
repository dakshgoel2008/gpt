import torch
import torch.nn as nn
from torchtyping import TensorType
from .transformer import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(model_dim, num_heads))
        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        # Token embeddings + positional embeddings
        embedded = self.word_embeddings(context)
        positions = torch.arange(context.shape[1], device=context.device)
        embedded = embedded + self.position_embeddings(positions)

        # Pass through N transformer blocks, then final LayerNorm
        output = self.final_norm(self.transformer_blocks(embedded))
        logits = self.vocab_projection(output)  # (B, T, vocab_size)

        return logits
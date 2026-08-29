import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gpt import GPT
from data.vocab import build_vocab, encode
from data.loader import create_batches

def train(model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        x, y = create_batches(data, context_length, batch_size)

        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return loss.item()

if __name__ == '__main__':
    # 1. Create a dummy dataset
    text = "hello world! this is a dummy text dataset to test the gpt training loop."
    text = text * 20  # Make it long enough for batches

    # 2. Build vocabulary
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    # 3. Encode data
    encoded_data = encode(text, stoi)
    data = torch.tensor(encoded_data, dtype=torch.long)

    # 4. Initialize model
    context_length = 8
    model_dim = 32
    num_blocks = 2
    num_heads = 4
    
    model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
    
    # 5. Train
    print("Starting training...")
    epochs = 500
    batch_size = 16
    lr = 1e-3
    train(model, data, epochs, context_length, batch_size, lr)

    # 6. Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'context_length': context_length,
        'model_dim': model_dim,
        'num_blocks': num_blocks,
        'num_heads': num_heads,
        'vocab_size': vocab_size
    }, 'model.pt')
    print("Model saved to model.pt")
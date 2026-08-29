import torch
import torch.nn as nn
from torchtyping import TensorType
from model.gpt import GPT
from data.vocab import encode, decode

def generate(model, new_chars: int, context: TensorType[int], context_length: int, itos: dict) -> str:
    result = []
    for _ in range(new_chars):
        if context.shape[1] > context_length:
            context = context[:, -context_length:]

        logits = model(context)
        last_logits = logits[:, -1, :]
        probs = nn.functional.softmax(last_logits, dim=-1)

        next_token = torch.multinomial(probs, 1)
        
        context = torch.cat((context, next_token), dim=-1)
        result.append(itos[next_token.item()])
        
    return ''.join(result)

if __name__ == '__main__':
    # 1. Load model checkpoint
    checkpoint = torch.load('model.pt')
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    model = GPT(
        vocab_size=checkpoint['vocab_size'],
        context_length=checkpoint['context_length'],
        model_dim=checkpoint['model_dim'],
        num_blocks=checkpoint['num_blocks'],
        num_heads=checkpoint['num_heads']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Setup context
    start_text = "hello"
    encoded_context = encode(start_text, stoi)
    context = torch.tensor([encoded_context], dtype=torch.long)

    # 3. Generate
    print(f"Generating from prompt: '{start_text}'")
    generated_text = generate(model, new_chars=50, context=context, context_length=checkpoint['context_length'], itos=itos)
    print(f"Generated text: {start_text}{generated_text}")
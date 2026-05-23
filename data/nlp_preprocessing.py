import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        sentences = positive + negative
        vocab = set()
        
        for i in sentences:
            for word in i.split():
                vocab.add(word)
        
        vocab = sorted(vocab)

        # word -> id
        word2id = {}
        for idx, word in enumerate(vocab, start = 1):
            word2id[word] = idx
        

        # Encode Sentences
        tensors = []

        for i in sentences:
            encoded = []
            for word in i.split():
                encoded.append(word2id[word])
            
            tensors.append(torch.tensor(encoded))

        # Padding of sequences
        padded = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first = True,
            padding_value = 0
        )

        return padded.float()
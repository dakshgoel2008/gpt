from typing import Dict, List, Tuple

def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    a=sorted(set(text))
    stoi={}
    itos={}
    for k,v in enumerate(a):
        itos[k]=v
        stoi[v]=k
    return (stoi,itos)

def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[c] for c in text]

def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return ''.join(itos[c] for c in ids)
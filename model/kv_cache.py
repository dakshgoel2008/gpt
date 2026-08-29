import torch
import torch.nn as nn
from typing import Tuple, Optional


class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat(
                [self.cache_k, new_k],
                dim=1
            )
            self.cache_v = torch.cat(
                [self.cache_v, new_v],
                dim=1
            )

        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()

        torch.manual_seed(0)

        self.q_proj = nn.Linear(
            model_dim,
            model_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            model_dim,
            model_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            model_dim,
            model_dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, KVCache]:

        # x: [B, T, D]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if kv_cache is None:
            kv_cache = KVCache()

        # Add new K and V to cache
        full_k, full_v = kv_cache.update(k, v)

        # [B, T, D] @ [B, D, S]
        # -> [B, T, S]
        scores = (
            q @ full_k.transpose(-2, -1)
        ) * (q.shape[-1] ** -0.5)

        # Causal mask
        T = q.shape[1]
        S = full_k.shape[1]

        # Query positions are the LAST T positions
        query_positions = torch.arange(
            S - T,
            S,
            device=x.device
        )

        key_positions = torch.arange(
            S,
            device=x.device
        )

        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        # [T, S] -> broadcast over batch
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ full_v

        return torch.round(output, decimals=4), kv_cache
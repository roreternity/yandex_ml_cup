import torch
import torch.nn as nn

from common import DENSE_DIM, VALUE_VOCAB

HIDDEN = 256
VALUE_EMB_DIM = 16

class PolicyNet(nn.Module):
    def __init__(self, num_actions: int, hidden: int = HIDDEN, value_emb_dim: int = VALUE_EMB_DIM):
        super().__init__()
        self.content_value_emb = nn.Embedding(VALUE_VOCAB, value_emb_dim)
        self.target_value_emb = nn.Embedding(VALUE_VOCAB, value_emb_dim)

        in_dim = DENSE_DIM + 2 * value_emb_dim
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, dense: torch.Tensor, content_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        cv = self.content_value_emb(content_value)
        tv = self.target_value_emb(target_value)
        x = torch.cat([dense, cv, tv], dim=-1)
        x = self.token_mlp(x)
        
        # Global pooling
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        
        # Logits over action space
        return self.head(pooled)
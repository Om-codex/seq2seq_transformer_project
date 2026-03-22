import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(           # Linear layer to add some non-linearity
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask = None):
        attn_out, _ = self.attn(x, mask) # applying multihead-attention

        x = self.norm1(x + attn_out) # first layer normalizing with residual connection (x + attn_out)

        ff_out = self.ff(x) # applying feed forward network to refine the representation

        x = self.norm2(x + ff_out) # second layer normalizing with residual connection (x + ff_out)

        return x # final output [batch, seq_len, d_model]   -> here each word is context-aware + refined


import torch
import torch.nn as nn
import torch.functional as F
from models.transformer import TransformerBlock

class MiniGPT(nn.Module):

    def __init__(self, vocab_size, d_model, heads, num_layers, max_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)

        x = self.embedding(x) + self.pos_embedding(positions)

        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.fc_out(x) 

        return logits

        
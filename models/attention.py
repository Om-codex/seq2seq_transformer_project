import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden]
        # encoder_outputs: [batch, seq_len, hidden]

        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.permute(1, 0, 2) # [batch, 1, hidden]
        hidden = hidden.repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) # [batch, seq_len, hidden]

        attention = self.v(energy).squeeze(2) # [batch, seq_len, 1]

        return F.softmax(attention, dim = 1)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads # the number of heads in the MultiHead Attention

        self.head_dim = d_model // num_heads # total num of dim in a single head

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask = None):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # splitting Q, K, & V into the heads
        # before split shape [batch, seq_len, d_model]

        batch_size = x.shape[0]

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim) # here pytorch (-1) calculates the remaining seq_len automatically
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)

        # after split shape [batch, seq_len, heads, head_dim]

        # now rearranging the shapes by permute
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # now the shape is [batch, heads, seq_len, head_dim]

        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores/ math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim = -1)

        context = torch.matmul(weights, V)

        context = context.permute(0, 2, 1, 3).contiguous() # combining the heads
        context = context.view(batch_size, -1, self.d_model)

        output = self.fc_out(context)

        return output, weights
    
    # Mutli-head attention lets the model look at the same sentence in multiple ways simultaneously like combining different brains to produce one output
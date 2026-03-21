import torch
import torch.nn as nn
import torch.nn.functional as f
from models.attention import Attention
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim + hidden_size, hidden_size, batch_first = True)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.attention = Attention(hidden_size)

    def forward(self, input_token, hidden, encoder_outputs):

        embedded = self.embedding(input_token)

        # Attention weights
        attn_weights = self.attention(hidden, encoder_outputs)

        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # Combine embedding + context
        rnn_input = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(rnn_input, hidden)

        prediction = self.fc(output)

        return prediction, hidden, attn_weights
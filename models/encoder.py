import torch
import torch.nn as nn

class Encoder(nn.Module):                                       #PyTorch neural network module

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)    #This layer converts word IDs → vectors

        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first = True)   #This is a type of RNN that processes sequences

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)

        outputs, hidden = self.gru(embedded)

        return outputs, hidden
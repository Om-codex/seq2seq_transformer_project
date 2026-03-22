import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5): # src -> source sequence (input language)   # trg -> target sequence (ground truth output)  # teacher_forcing_ratio -> probability of using the true previous token instead of predicted token
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input_token = trg[:,0].unsqueeze(1)

        attn_weights_all = []

        for t in range(1, trg_len):

            output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)

            outputs[:,t] = output.squeeze(1)

            top1 = output.argmax(2)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            input_token = trg[:,t].unsqueeze(1) if teacher_force else top1

            attn_weights_all.append(attn_weights)

        return outputs, attn_weights_all
import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):

    def __init__(self, sentences, vocab, seq_len):

        self.vocab = vocab
        self.seq_len = seq_len
        self.data = []

        for sentence in sentences:
            tokens = [vocab.word2index.get(w, vocab.word2index["UNK"])
                      for w in sentence.lower().split()]

            tokens = [vocab.word2index["SOS"]] + tokens + [vocab.word2index["EOS"]]

            # sliding window
            for i in range(len(tokens) - seq_len):
                self.data.append(tokens[i:i+seq_len+1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        seq = self.data[idx]

        x = torch.tensor(seq[:-1])
        y = torch.tensor(seq[1:])

        return x, y
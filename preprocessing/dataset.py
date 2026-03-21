from torch.utils.data import Dataset
from preprocessing.sentence_to_tensor import sentence_to_tensor
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 2 # since the vocabulary we build has PAD = 2 index

class TranslationDataset(Dataset):

    def __init__(self, pairs, eng_vocab, fr_vocab):

        self.pairs = pairs
        self.eng_vocab = eng_vocab
        self.fr_vocab = fr_vocab

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        eng, fr = self.pairs[idx]

        src = sentence_to_tensor(self.eng_vocab, eng)
        trg = sentence_to_tensor(self.fr_vocab, fr)

        return src, trg

def collate_fn(batch):

    src_batch, trg_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)

    return src_batch, trg_batch
import torch
def sentence_to_tensor(vocab, sentence):

    tokens = []

    for word in sentence.lower().split():

        if word in vocab.word2index:
            tokens.append(vocab.word2index[word])

        else:
            tokens.append(vocab.word2index["UNK"])

    tokens.insert(0, vocab.word2index["SOS"])
    tokens.append(vocab.word2index["EOS"])
    return torch.tensor(tokens, dtype = torch.long)
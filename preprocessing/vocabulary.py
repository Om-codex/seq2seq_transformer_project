from collections import Counter

class Vocabulary:
    def __init__(self, name):
        self.name = name

        self.word2index = {
            "SOS": 0,
            "EOS": 1,
            "PAD": 2,
            "UNK": 3
        }

        self.index2word = {
            0: "SOS",
            1: "EOS",
            2: "PAD",
            3: "UNK"
        }

        self.word_count = Counter()
        self.vocab_size = 4

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            self.word_count[word] += 1

    def build_vocab(self, max_size):
        # sort words by frequency
        most_common = self.word_count.most_common(max_size)

        for word, count in most_common:
            self.word2index[word] = self.vocab_size
            self.index2word[self.vocab_size] = word
            self.vocab_size += 1
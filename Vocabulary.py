from collections import Counter


class Vocabulary:
    def __init__(self):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx = 4

    def add_char(self, char):
        if char not in self.stoi:
            self.stoi[char] = self.idx
            self.itos[self.idx] = char
            self.idx += 1

    def build_vocabulary(self, labels):
        char_counter = Counter(char for label in labels for char in label)
        for char, _ in char_counter.items():
            self.add_char(char)

    def __len__(self):
        return len(self.itos)

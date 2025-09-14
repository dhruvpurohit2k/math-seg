from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import torch


class CrohmeDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, vocab=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(labels_file, sep="\t")
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        image = Image.open(img_path).convert("L")
        label = self.df.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        numerical_label = [self.vocab.stoi["<start>"]]
        numerical_label += [
            self.vocab.stoi.get(char, self.vocab.stoi["<unk>"]) for char in label
        ]
        numerical_label.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numerical_label)

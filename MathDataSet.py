import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class MathDataset(Dataset):
    def __init__(self, numpy_file, vocab):
        # Load the preprocessed data directly into RAM
        data = np.load(numpy_file, allow_pickle=True)
        self.images = data["images"]
        self.labels = data["labels"]
        self.vocab = vocab
        self.transform = transforms.ToTensor()  # Only need to convert to tensor now

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Access data directly from RAM - much faster than disk I/O
        image_np = self.images[idx]
        label_str = self.labels[idx]

        # Convert numpy array to tensor
        image = self.transform(image_np)

        # Numericalize the label
        numerical_label = [self.vocab.stoi["<start>"]]
        numerical_label.extend(self.vocab.numericalize(label_str))
        numerical_label.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numerical_label)

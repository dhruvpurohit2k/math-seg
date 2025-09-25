import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from collections import Counter
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR  # --- NEW: Import LR Scheduler ---
from tqdm import tqdm


# --- Disable Pillow's DecompressionBombWarning ---
Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
IMG_HEIGHT = 96
NUM_EPOCHS = 30  # --- IMPROVEMENT 4: Train for longer ---
LEARNING_RATE = 1e-4
ACCUMULATION_STEPS = 2
TRAIN_ON_SUBSET = False
SUBSET_PERCENT = 0.1

# Model Dimensions
EMBED_DIM = 256
ATTENTION_DIM = 256
DECODER_DIM = 512
ENCODER_DIM = 512
DROPOUT = 0.5  # --- IMPROVEMENT 3: Add Dropout ---

print(f"Using device: {DEVICE}")
print(f"Actual Batch Size: {BATCH_SIZE}")

ENCODER_PATH = "./encoder.pth"
DECODER_PATH = "./decoder.pth"
VOCAB_PATH = "./vocab.pth"
NUMPY_FILE = "./data/preprocessed_data.npz"
BASE_PATH = "."
LABELS_FILE = "./data/labels.tsv"
OUTPUT_IMAGE_DIR = "./data/generated_images/"


# --- 1. Vocabulary ---
class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for char in sentence:
                frequencies[char] += 1
        for char, count in frequencies.items():
            if count >= self.freq_threshold:
                if char not in self.stoi:
                    self.stoi[char] = len(self.stoi)
                    self.itos[len(self.itos)] = char

    def numericalize(self, text):
        return [self.stoi.get(char, self.stoi["<unk>"]) for char in text]


# --- 2. Memory-Efficient Custom Dataset (Unchanged) ---
class MathDataset(Dataset):
    def __init__(self, image_dir, labels_file, vocab, transforms):
        self.df = pd.read_csv(labels_file, sep="\t")
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_filename"])
        label_str = row["latex_label"]

        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            print(f"Warning: Image not found: {img_path}. Returning a dummy image.")
            image = Image.new("L", (100, IMG_HEIGHT), color=255)

        image = self.transform(image)

        numerical_label = [self.vocab.stoi["<start>"]]
        numerical_label.extend(self.vocab.numericalize(label_str))
        numerical_label.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numerical_label)


# --- 3. Custom Collate Function for Padding (Unchanged) ---
class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        max_width = max([img.shape[2] for img in imgs])
        padded_imgs = [
            F.pad(img, (0, max_width - img.shape[2]), "constant", 1.0) for img in imgs
        ]
        targets = nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )
        return torch.stack(padded_imgs), targets


# --- 4. Model Architecture (WITH DROPOUT) ---
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, ENCODER_DIM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ENCODER_DIM),
            nn.ReLU(),
        )

    def forward(self, images):
        return self.cnn(images)


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=ENCODER_DIM,
        dropout=DROPOUT,
    ):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(
            p=dropout
        )  # --- IMPROVEMENT 3: Initialize Dropout ---

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(2) * encoder_out.size(3)
        encoder_out = encoder_out.permute(0, 2, 3, 1).reshape(
            batch_size, num_pixels, self.encoder_dim
        )
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = [cl - 1 for cl in caption_lengths]
        predictions = torch.zeros(
            batch_size, max(decode_lengths), self.fc.out_features
        ).to(DEVICE)
        for t in range(max(decode_lengths)):
            batch_size_t = sum(l > t for l in decode_lengths)
            awe, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            awe = gate * awe
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], awe], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))  # --- IMPROVEMENT 3: Apply Dropout ---
            predictions[:batch_size_t, t, :] = preds
        return predictions, encoded_captions, decode_lengths, None


# --- 5. Training Loop (WITH IMPROVEMENTS) ---
def train():
    print("Loading data and building vocabulary...")

    df = pd.read_csv(LABELS_FILE, sep="\t")
    vocab = Vocabulary()
    vocab.build_vocabulary(df["latex_label"].tolist())

    print(f"Vocabulary size: {len(vocab)}")

    image_transforms = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.resize(
                    (int(image.width * IMG_HEIGHT / image.height), IMG_HEIGHT)
                )
            ),
            transforms.ToTensor(),
        ]
    )

    dataset = MathDataset(
        image_dir=OUTPUT_IMAGE_DIR,
        labels_file=LABELS_FILE,
        vocab=vocab,
        transforms=image_transforms,
    )

    if TRAIN_ON_SUBSET:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        subset_size = int(dataset_size * SUBSET_PERCENT)
        np.random.shuffle(indices)
        subset_indices = indices[:subset_size]
        dataset = Subset(dataset, subset_indices)
        print(
            f"--- Training on a subset of {len(dataset)} images ({SUBSET_PERCENT*100}%) ---"
        )

    pad_idx = vocab.stoi["<pad>"]
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=PadCollate(pad_idx=pad_idx),
        num_workers=2,
    )

    encoder = Encoder().to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        params=list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # --- IMPROVEMENT 1: Initialize the Learning Rate Scheduler ---
    # This will decrease the learning rate by 10% every 5 epochs.
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    print("Initializing model...")
    print("Starting training...")
    encoder.train()
    decoder.train()

    for epoch in range(NUM_EPOCHS):
        for i, (imgs, captions) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        ):
            imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)

            features = encoder(imgs)
            outputs, _, decode_lengths, _ = decoder(
                features, captions, [len(c) for c in captions]
            )
            targets = captions[:, 1:]

            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, decode_lengths, batch_first=True
            ).data
            targets = nn.utils.rnn.pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = criterion(outputs, targets)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # --- IMPROVEMENT 2: Add Gradient Clipping ---
            # This must be done *after* loss.backward() but *before* optimizer.step()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        # --- IMPROVEMENT 1: Step the scheduler at the end of each epoch ---
        scheduler.step()
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Final Batch Loss: {loss.item() * ACCUMULATION_STEPS:.4f}, LR: {scheduler.get_last_lr()[0]}"
        )

    print("Training complete.")
    print("Saving model weights and vocabulary...")
    torch.save(encoder.state_dict(), ENCODER_PATH)
    torch.save(decoder.state_dict(), DECODER_PATH)
    torch.save(vocab, VOCAB_PATH)


# --- Start training ---
train()

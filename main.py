import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import pandas as pd
import os
import torch.nn as nn


from Encoder import Encoder
from DecoderWithAttention import DecoderWithAttention
from Vocabulary import Vocabulary
from CrohmeDataset import CrohmeDataset
from Collate import Collate


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    encoder, decoder = model
    encoder.train()
    decoder.train()

    total_loss = 0
    for imgs, labels, lengths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        features = encoder(imgs)
        predictions, targets, decode_lengths = decoder(features, labels, lengths)

        # To calculate loss, we need to remove the <start> token from targets
        # and match the dimensions of predictions.
        targets = targets[:, 1:]

        # pack_padded_sequence is a good way to handle variable lengths
        # and ignore padded elements in loss calculation
        predictions_packed = torch.nn.utils.rnn.pack_padded_sequence(
            predictions, decode_lengths, batch_first=True, enforce_sorted=False
        ).data
        targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
            targets, decode_lengths, batch_first=True, enforce_sorted=False
        ).data

        loss = loss_fn(predictions_packed, targets_packed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


Image.MAX_IMAGE_PIXELS = None
if __name__ == "__main__":
    # --- 4. Setup & Configuration ---
    # Create dummy data if it doesn't exist for demonstration
    if not os.path.exists("./data/images"):
        os.makedirs("./data/images")
        dummy_img = Image.new("L", (200, 96), "white")
        draw = ImageDraw.Draw(dummy_img)
        draw.text((10, 40), "1+1", fill="black")
        dummy_img.save("./data/images/dummy.png")
        with open("./data/labels.csv", "w") as f:
            f.write("image_filename,latex_label\n")
            f.write("dummy.png,1+1\n")

    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    IMG_HEIGHT = 96

    # Model dimensions
    EMBED_DIM = 256
    ATTENTION_DIM = 256
    DECODER_DIM = 512

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 5. Data Loading and Preparation ---
    print("Loading data and building vocabulary...")
    labels_df = pd.read_csv("./data/labels.tsv", sep="\t")

    vocab = Vocabulary()
    vocab.build_vocabulary(labels_df["latex_label"].tolist())
    print(f"Vocabulary size: {len(vocab)}")

    image_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, 512)),  # Resize to a large fixed size first
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = CrohmeDataset(
        root_dir="./data/images",
        labels_file="./data/labels.tsv",
        transform=image_transforms,
        vocab=vocab,
    )

    pad_idx = vocab.stoi["<pad>"]
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Collate(pad_idx=pad_idx, target_height=IMG_HEIGHT),
    )

    # --- 6. Model, Optimizer, and Loss ---
    print("Initializing model...")
    encoder = Encoder().to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
    ).to(DEVICE)

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # --- 7. Main Training Loop ---
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(
            model=(encoder, decoder),
            loader=data_loader,
            optimizer=optimizer,
            loss_fn=criterion,
            device=DEVICE,
        )
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    print("Training complete.")
    # Here you would typically save your model's weights
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
    torch.save(vocab, "vocab.pth")

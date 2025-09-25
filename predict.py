import torch
from PIL import Image, ImageDraw
import os
from torchvision.transforms import transforms
from Encoder import Encoder
from DecoderWithAttention import DecoderWithAttention
from Vocabulary import Vocabulary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
IMG_HEIGHT = 96  # Must match value in Cell 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
ACCUMULATION_STEPS = 2

# Model Dimensions

EMBED_DIM = 256
ATTENTION_DIM = 256
DECODER_DIM = 512
ENCODER_DIM = 512

print(f"Using device: {DEVICE}")
print(f"Actual Batch Size: {BATCH_SIZE}")
print(f"Effective Batch Size (with accumulation): {BATCH_SIZE * ACCUMULATION_STEPS}")

## PATHS
ENCODER_PATH = "./encoder.pth"
DECODER_PATH = "./decoder.pth"
VOCAB_PATH = "./vocab.pth"
NUMPY_FILE = "./data/preprocessed_data.npz"
BASE_PATH = "."


def predict(encoder, decoder, image_path, vocab, device, transforms, max_length=150):
    encoder.eval()
    decoder.eval()

    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    image = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)
        num_pixels = features.size(2) * features.size(3)
        features = features.permute(0, 2, 3, 1).reshape(
            1, num_pixels, decoder.encoder_dim
        )

    predicted_sequence = [vocab.stoi["<start>"]]
    h, c = decoder.init_hidden_state(features)

    for _ in range(max_length):
        previous_word = torch.tensor([predicted_sequence[-1]]).to(device)
        embeddings = decoder.embedding(previous_word)

        awe, _ = decoder.attention(features, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        output = decoder.fc(h)
        predicted_token = output.argmax(1).item()

        predicted_sequence.append(predicted_token)
        if predicted_token == vocab.stoi["<end>"]:
            break

    return "".join([vocab.itos[idx] for idx in predicted_sequence[1:-1]])


def run_inference():
    print("--- Running Inference Example ---")

    prediction_image_path = os.path.join(BASE_PATH, "onepone.png")
    if not os.path.exists(prediction_image_path):
        print(f"Creating a dummy image for prediction at: {prediction_image_path}")
        pred_img = Image.new("L", (400, 100), color=255)
        draw = ImageDraw.Draw(pred_img)
        draw.text((10, 30), "x^2 + y^2 = z^2", fill="black")
        pred_img.save(prediction_image_path)

    print("Loading trained model weights and vocabulary...")
    try:
        vocab_pred = torch.load(VOCAB_PATH, weights_only=False)
    except FileNotFoundError:
        print("Vocabulary file not found. Please train the model first.")
        return

    encoder_pred = Encoder().to(DEVICE)
    decoder_pred = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab_pred),
    ).to(DEVICE)

    try:
        encoder_pred.load_state_dict(torch.load(ENCODER_PATH))
        decoder_pred.load_state_dict(torch.load(DECODER_PATH))
    except FileNotFoundError:
        print("Model weights not found. Please train the model first.")
        return

    # Use the same transforms as training for consistency
    pred_transforms = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.resize(
                    (int(image.width * IMG_HEIGHT / image.height), IMG_HEIGHT)
                )
            ),
            transforms.ToTensor(),
        ]
    )

    predicted_latex = predict(
        encoder=encoder_pred,
        decoder=decoder_pred,
        image_path=prediction_image_path,
        vocab=vocab_pred,
        device=DEVICE,
        transforms=pred_transforms,
    )

    if predicted_latex is not None:
        print(f"\nImage Path: {prediction_image_path}")
        print(f"Predicted LaTeX: {predicted_latex}")


# --- Run inference ---
run_inference()
print("Inference cell is ready. Uncomment 'run_inference()' to predict on an image.")

import torch
from PIL import Image, ImageDraw
import os
from torchvision.transforms import transforms
from Encoder import Encoder
from DecoderWithAttention import DecoderWithAttention
from Vocabulary import Vocabulary

EMBED_DIM = 256
ATTENTION_DIM = 256
DECODER_DIM = 512
IMG_HEIGHT = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(encoder, decoder, image_path, vocab, device, transforms, max_length=150):
    """
    Takes a trained model and an image, and returns the predicted LaTeX string.
    """
    encoder.eval()
    decoder.eval()

    # 1. Load and preprocess the image
    try:
        image = Image.open(image_path).convert("L")
        image = transforms(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    # 2. Pass image through the encoder
    with torch.no_grad():
        features = encoder(image)
        num_pixels = features.size(2) * features.size(3)
        features = features.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        features = features.view(1, num_pixels, decoder.encoder_dim)

    # 3. Auto-regressive decoding loop
    predicted_sequence = [vocab.stoi["<start>"]]

    # Initialize decoder hidden state
    h, c = decoder.init_hidden_state(features)

    for t in range(max_length):
        # Get the last predicted token
        previous_token = torch.LongTensor([predicted_sequence[-1]]).to(device)

        with torch.no_grad():
            # Get embeddings and context vector
            embeddings = decoder.embedding(previous_token)
            attention_weighted_encoding, _ = decoder.attention(features, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Run through LSTM cell
            lstm_input = torch.cat((embeddings, attention_weighted_encoding), dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))

            # Get prediction
            output = decoder.fc(h)
            top_prediction = output.argmax(1)

        predicted_sequence.append(top_prediction.item())

        # Stop if we predict the <end> token
        if top_prediction.item() == vocab.stoi["<end>"]:
            break

    # 4. Convert token IDs back to characters
    output_latex = []
    for token_idx in predicted_sequence:
        # Skip <start> token in the final output
        if token_idx == vocab.stoi["<start>"]:
            continue
        # Stop at <end> token
        if token_idx == vocab.stoi["<end>"]:
            break
        output_latex.append(vocab.itos[token_idx])

    return "".join(output_latex)


# Create a dummy image for prediction if it doesn't exist
prediction_image_path = "./example/avg.png"
if not os.path.exists(prediction_image_path):
    print(f"Creating a dummy image for prediction at: {prediction_image_path}")
    pred_img = Image.new("L", (400, 96), "white")
    draw = ImageDraw.Draw(pred_img)
    # Using a simple text draw, but in reality, this would be your handwritten image
    draw.text((10, 30), "x^2 + y^2 = z^2", fill="black", font_size=30)
    pred_img.save(prediction_image_path)

print("Loading trained model weights...")

vocab = torch.load("vocab.pth", weights_only=False)
encoder_pred = Encoder().to(DEVICE)
decoder_pred = DecoderWithAttention(
    attention_dim=ATTENTION_DIM,
    embed_dim=EMBED_DIM,
    decoder_dim=DECODER_DIM,
    vocab_size=len(vocab),
).to(DEVICE)

encoder_pred.load_state_dict(torch.load("encoder.pth"))
decoder_pred.load_state_dict(torch.load("decoder.pth"))

image_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, 512)),  # Resize to a large fixed size first
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# 2. Run the prediction function
predicted_latex = predict(
    encoder=encoder_pred,
    decoder=decoder_pred,
    image_path=prediction_image_path,
    vocab=vocab,
    device=DEVICE,
    transforms=image_transforms,
)

if predicted_latex is not None:
    print(f"\nImage Path: {prediction_image_path}")
    print(f"Predicted LaTeX: {predicted_latex}")

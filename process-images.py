import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm  # For a nice progress bar
import os

# --- Configuration from Training Cell (must be consistent) ---
IMG_HEIGHT = 96

BASE_PATH = "./"

Image.MAX_IMAGE_PIXELS = None
# Define paths for all your data and model files
INKML_INPUT_DIR = os.path.join(BASE_PATH, "dataset/crohme2019/crohme2019/train")
OUTPUT_IMAGE_DIR = os.path.join(BASE_PATH, "data/generated_images")
LABELS_FILE = os.path.join(BASE_PATH, "data/labels.tsv")
# --- NEW: Path for the preprocessed NumPy file ---
NUMPY_FILE = os.path.join(BASE_PATH, "data/preprocessed_data.npz")


MODEL_DIR = os.path.join(BASE_PATH, "models")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pth")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pth")

# Create the necessary directories if they don't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Google Drive mounted and all paths are set up.")
print(f"Your images will be saved in: {OUTPUT_IMAGE_DIR}")
print(f"Your preprocessed data will be saved as: {NUMPY_FILE}")
print(f"Your models will be saved in: {MODEL_DIR}")


def parse_inkml_to_image(inkml_file_path, output_dir):
    """Parses a single .inkml file, saves it as an image, and returns the LaTeX label."""
    try:
        tree = ET.parse(inkml_file_path)
        root = tree.getroot()

        namespace = "{http://www.w3.org/2003/InkML}"
        truth_annotation = root.find(f'{namespace}annotation[@type="truth"]')
        if truth_annotation is None or not truth_annotation.text:
            return None, None

        latex_label = truth_annotation.text.strip()

        traces = root.findall(f"{namespace}trace")
        if not traces:
            return None, None

        points = []
        for trace in traces:
            coords = trace.text.strip().split(",")
            trace_points = [
                (float(p.strip().split()[0]), float(p.strip().split()[1]))
                for p in coords
                if p.strip() and len(p.strip().split()) >= 2
            ]
            points.append(trace_points)

        if not any(points):
            return None, None

        all_x = [p[0] for trace in points for p in trace]
        all_y = [p[1] for trace in points for p in trace]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        padding = 20
        width = int(max_x - min_x + 2 * padding)
        height = int(max_y - min_y + 2 * padding)

        if width <= 0 or height <= 0:
            return None, None

        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)

        for trace in points:
            if len(trace) > 1:
                adjusted_trace = [
                    (p[0] - min_x + padding, p[1] - min_y + padding) for p in trace
                ]
                draw.line(adjusted_trace, fill=0, width=3)

        base_filename = os.path.splitext(os.path.basename(inkml_file_path))[0]
        image_filename = f"{base_filename}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)

        return image_filename, latex_label

    except Exception as e:
        print(f"Error processing {inkml_file_path}: {e}")
        return None, None


def process_all_inkml_files(input_dir, output_dir, labels_file):
    """Processes all .inkml files into PNGs and creates a labels.tsv file."""
    print("Starting to process .inkml files into PNGs...")
    inkml_files = glob.glob(os.path.join(input_dir, "*.inkml"))

    with open(labels_file, "w") as f:
        f.write("image_filename\tlatex_label\n")
        for i, inkml_file in enumerate(tqdm(inkml_files, desc="Parsing InkML")):
            image_filename, latex_label = parse_inkml_to_image(inkml_file, output_dir)
            if image_filename and latex_label:
                f.write(f"{image_filename}\t{latex_label}\n")

    print(f"PNG processing complete.")


def preprocess_and_save_numpy(image_dir, labels_file, output_numpy_file):
    """Loads images, preprocesses them, and saves them to a single .npz file."""
    print(f"\nStarting preprocessing of images into NumPy file: {output_numpy_file}")
    df = pd.read_csv(labels_file, sep="\t")

    processed_images = []
    processed_labels = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Images"):
        img_path = os.path.join(image_dir, row["image_filename"])
        try:
            image = Image.open(img_path).convert("L")
            w, h = image.size
            if h == 0:
                continue
            new_w = int(w * IMG_HEIGHT / h)

            if new_w <= 0:
                continue

            image = image.resize((new_w, IMG_HEIGHT))
            image_np = np.array(image)
            # --- THE  FIX: Forcefully reshape the array to guarantee it is 2D. ---

            # This is more direct than checking dimensions and handles all edge cases.
            if image_np.shape[1] < 2:
                print(image_np.shape)
            image_np = image_np.reshape((IMG_HEIGHT, new_w))
            if image_np.ndim != 2:
                # This will print a very obvious message if it finds a bad image.
                print("\n" + "=" * 80)
                print(
                    f"[!!!!] DEBUG: Found a non-2D array for image '{row['image_filename']}'."
                )
                print(f"[!!!!] Shape was {image_np.shape}. THIS IMAGE WILL BE SKIPPED.")
                print("=" * 80 + "\n")
                continue  # Skip this corrupted/malformed image

            processed_images.append(image_np)
            processed_labels.append(row["latex_label"])
        except FileNotFoundError:
            pass
        except Exception as e:
            # Added more detail to the error message for better debugging
            print(
                f"\n[Error] Could not process image {img_path}. Shape after resize: {(IMG_HEIGHT, new_w)}. Error: {e}"
            )
    # SOLUTION 1: Create object array properly to avoid broadcasting error
    images_array = np.empty(len(processed_images), dtype=object)
    for i, img in enumerate(processed_images):
        images_array[i] = img

    np.savez_compressed(
        output_numpy_file,
        images=images_array,
        labels=np.array(processed_labels),
    )
    print(f"Preprocessing complete. Data saved to {output_numpy_file}")


# process_all_inkml_files(INKML_INPUT_DIR, OUTPUT_IMAGE_DIR, LABELS_FILE)
preprocess_and_save_numpy(OUTPUT_IMAGE_DIR, LABELS_FILE, NUMPY_FILE)

print(
    "Parser cell is ready. Uncomment the function calls to run the full data processing pipeline."
)

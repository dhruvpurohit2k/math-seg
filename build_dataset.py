import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# --- Configuration ---
# Directory where your CROHME .inkml files are located
INKML_INPUT_DIR = "./dataset/crohme2019/crohme2019/train/"
# Directory to save the generated PNG images
OUTPUT_IMAGE_DIR = "./data/images"
# The name of the output labels file
OUTPUT_LABELS_FILE = "./data/labels.tsv"


# --- Main Script ---
def parse_inkml_to_images_and_tsv():
    """
    Parses all .inkml files in the input directory, renders them as PNG images,
    and creates a tab-separated (TSV) file mapping filenames to LaTeX labels.
    """
    # 1. Create output directories if they don't exist
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"Created directory: {OUTPUT_IMAGE_DIR}")

    # Create a dummy input directory and file if it doesn't exist for demonstration
    if not os.path.exists(INKML_INPUT_DIR):
        os.makedirs(INKML_INPUT_DIR)
        dummy_inkml_content = """
<ink xmlns="http://www.w3.org/2003/InkML">
<traceGroup>
<trace>100 100, 110 100, 120 105</trace>
<trace>110 90, 110 110</trace>
</traceGroup>
<annotation type="truth">1+1</annotation>
</ink>
"""
        with open(os.path.join(INKML_INPUT_DIR, "dummy_equation.inkml"), "w") as f:
            f.write(dummy_inkml_content)
        print(f"Created dummy inkml file for demonstration.")

    # 2. Open the output file for writing
    with open(OUTPUT_LABELS_FILE, "w", encoding="utf-8") as labels_file:
        # Write the header row
        labels_file.write("image_filename\tlatex_label\n")

        print(f"Starting parsing of .inkml files from: {INKML_INPUT_DIR}")

        # 3. Iterate over all files in the input directory
        for filename in os.listdir(INKML_INPUT_DIR):
            if not filename.endswith(".inkml"):
                continue

            inkml_path = os.path.join(INKML_INPUT_DIR, filename)

            try:
                # 4. Parse the XML structure of the .inkml file
                tree = ET.parse(inkml_path)
                root = tree.getroot()

                # Namespace is often present in these files
                namespace = {"inkml": "http://www.w3.org/2003/InkML"}

                # 5. Extract the LaTeX label (ground truth)
                truth_element = root.find('inkml:annotation[@type="truth"]', namespace)
                if truth_element is None or truth_element.text is None:
                    print(f"Warning: No truth label found in {filename}. Skipping.")
                    continue
                latex_label = truth_element.text.strip()

                # 6. Extract all stroke coordinates
                traces = []
                for trace in root.findall(".//inkml:trace", namespace):
                    points = trace.text.strip().split(",")
                    stroke = []
                    for point in points:
                        coords = point.strip().split()
                        if len(coords) == 2:
                            stroke.append((float(coords[0]), float(coords[1])))
                    if stroke:
                        traces.append(stroke)

                if not traces:
                    print(f"Warning: No traces found in {filename}. Skipping.")
                    continue

                # 7. Render the strokes onto a PNG image
                # Find the bounding box of the strokes to size the image correctly
                all_x = [p[0] for s in traces for p in s]
                all_y = [p[1] for s in traces for p in s]
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)

                # Add some padding
                padding = 20
                img_width = int(max_x - min_x + 2 * padding)
                img_height = int(max_y - min_y + 2 * padding)

                # Create a new white image
                image = Image.new("L", (img_width, img_height), "white")
                draw = ImageDraw.Draw(image)

                # Draw each stroke
                for stroke in traces:
                    # Normalize coordinates to fit the image
                    normalized_stroke = [
                        (x - min_x + padding, y - min_y + padding) for x, y in stroke
                    ]
                    draw.line(normalized_stroke, fill="black", width=3)

                # 8. Save the image and write to the TSV file
                image_filename = os.path.splitext(filename)[0] + ".png"
                image.save(os.path.join(OUTPUT_IMAGE_DIR, image_filename))

                # IMPORTANT: Write with a tab separator
                labels_file.write(f"{image_filename}\t{latex_label}\n")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print("\nProcessing complete.")
    print(f"Images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"Labels saved to: {OUTPUT_LABELS_FILE}")


if __name__ == "__main__":
    parse_inkml_to_images_and_tsv()

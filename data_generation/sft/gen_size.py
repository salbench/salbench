import argparse
import random
import math
import multiprocessing
from PIL import Image, ImageDraw, ImageColor
import io
import json
import tarfile
import os
import tempfile
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from reportlab.lib.colors import Color
from tqdm import tqdm
from glob import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate size variation image dataset")
    parser.add_argument("--main_path", type=str, default="saliency_data_sft_v2/size", help="Path to save dataset")
    parser.add_argument("--total_images", type=int, default=166000, help="Total number of images to generate")
    parser.add_argument("--images_per_folder", type=int, default=1000, help="Number of images per folder")
    parser.add_argument("--tokenizer_model", type=str, default="gpt2", help="Tokenizer model to use")
    return parser.parse_args()


def load_and_color_icon(icon_path, max_size, color):
    drawing = svg2rlg(icon_path)

    # Apply color to the icon
    def apply_color(shape):
        if hasattr(shape, 'fillColor'):
            shape.fillColor = Color(color[0]/255, color[1]/255, color[2]/255, alpha=1)
        if hasattr(shape, 'strokeColor'):
            shape.strokeColor = Color(color[0]/255, color[1]/255, color[2]/255, alpha=1)
        if hasattr(shape, 'strokeWidth') and shape.strokeWidth > 0:
            shape.strokeWidth = 0.1  # Ensure the stroke is visible if it's meant to be

        if hasattr(shape, 'children'):
            for child in shape.children:
                apply_color(child)

    apply_color(drawing)
    # Render to PIL Image
    buf = io.BytesIO()
    renderPM.drawToFile(drawing, buf, fmt="PNG")
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    img = img.resize((max_size, max_size), Image.LANCZOS)

    return img


def generate_size_image(args):
    icon_path, size, grid_size, background_color = args
    img = Image.new('RGB', (size, size), background_color)

    cell_size = size // grid_size
    base_icon_size = int(cell_size * 0.8)

    # Random color for this image
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # Size ratio for the target icon (from 0.8 to 1.5 times the base size)
    size_ratio = random.uniform(0.8, 1.5)
    target_icon_size = int(base_icon_size * size_ratio)

    max_icon_size = max(base_icon_size, target_icon_size)

    # Load and color the icon at maximum required size
    icon = load_and_color_icon(icon_path, max_icon_size, color)

    # Vary the number of objects
    min_objects = (grid_size * grid_size) // 2
    max_objects = grid_size * grid_size
    num_objects = random.randint(min_objects, max_objects)

    # Randomly select positions for icons
    positions = random.sample(
        [(x, y) for x in range(grid_size) for y in range(grid_size)],
        num_objects
    )

    # Select target position
    target_pos = random.choice(positions)

    for x, y in positions:
        if (x, y) == target_pos:
            # Use target icon size
            current_icon_size = target_icon_size
        else:
            current_icon_size = base_icon_size

        # Resize icon accordingly
        current_icon = icon.resize(
            (current_icon_size, current_icon_size),
            Image.LANCZOS
        )

        paste_x = x * cell_size + (cell_size - current_icon_size) // 2
        paste_y = y * cell_size + (cell_size - current_icon_size) // 2

        img.paste(current_icon, (paste_x, paste_y), current_icon)

    return img, os.path.basename(icon_path).split('.')[0], target_pos, size_ratio, num_objects, color

def generate_conversation_for_icon(shape, target_pos, size_ratio, grid_size, num_objects):
    """
    Generate a conversation about an image based on its attributes.

    Parameters:
        shape (str): The shape of the objects (e.g., "square", "icon").
        target_pos (tuple): The (x, y) position of the unique-sized object in the grid.
        size_ratio (float): Ratio indicating the size difference of the unique object.
        grid_size (int): Dimension of the grid (e.g., 7x7).
        num_objects (int): Total number of icons in the grid.

    Returns:
        list of tuples: A conversation as a list of question-answer pairs.
    """
    
    # Convert target_pos to normalized coordinates
    norm_x = round((target_pos[0] + 0.5) / grid_size, 2)
    norm_y = round((target_pos[1] + 0.5) / grid_size, 2)

    # Determine size description based on size_ratio
    if size_ratio < 1.0:
        size_description = "smaller" if size_ratio < 0.9 else "slightly smaller"
    else:
        size_description = "bigger" if size_ratio > 1.2 else "slightly bigger"

    # Question and Answer templates
    question_templates = {
        "num_objects": [
            "How many objects are in this image?",
            "What is the total number of icons displayed here?",
            "Can you tell me the count of icons in this image?",
            "How many shapes are present in the grid?"
        ],
        "shape_type": [
            "What shapes are these objects?",
            "Can you describe the shape of each icon?",
            "What type of shape is used for these objects?",
            "Are all the icons the same shape?"
        ],
        "unique_object_size": [
            "Is there any icon that has a different size?",
            "Do you notice any icon that is larger or smaller than the others?",
            "Is there an icon that stands out due to its size?",
            "Are all the icons the same size, or is one different?"
        ],
        "unique_object_position": [
            "Where is the differently-sized icon located?",
            "Can you tell me the position of the unique-sized icon?",
            "Where would I find the icon with a different size in the grid?",
            "Can you specify the location of the unique-sized icon?"
        ],
        "difference_description": [
            "How is the unique icon different from the others?",
            "What makes one of the icons stand out from the rest?",
            "Is there something unique about one of the icons?",
            "How does the unique icon differ from its counterparts?"
        ]
    }

    answer_templates = {
        "num_objects": [
            f"There are {num_objects} icons in this image.",
            f"The image contains {num_objects} icons arranged in a grid.",
            f"A total of {num_objects} icons can be seen here.",
            f"There are exactly {num_objects} objects in the grid."
        ],
        "shape_type": [
            f"Each icon is shaped like a {shape}.",
            f"All icons in the image are {shape}-shaped.",
            f"The objects in this image are all {shape}s.",
            f"Each shape in the grid is a {shape}."
        ],
        "unique_object_size": [
            "Yes, one icon has a unique size compared to the others.",
            "There is one icon that differs in size.",
            "One icon stands out as it is of a different size.",
            "Indeed, one of the icons is uniquely sized."
        ],
        "unique_object_position": [
            f"The unique-sized icon is located at normalized coordinates ({norm_x}, {norm_y}).",
            f"You'll find the differently-sized icon at the coordinates ({norm_x}, {norm_y}).",
            f"The distinct icon is positioned around ({norm_x}, {norm_y}) in normalized coordinates.",
            f"The icon with a unique size is near ({norm_x}, {norm_y}) in the normalized grid."
        ],
        "difference_description": [
            f"The unique icon is {size_description} than the others.",
            f"This icon differs from the others in that it is {size_description}.",
            f"The distinguishing feature of this icon is its size, as it is {size_description}.",
            f"It stands out by being {size_description} compared to the rest."
        ]
    }

    # Select random questions for diversity
    selected_question_types = random.sample(["num_objects", "shape_type", "unique_object_size", "unique_object_position"], 4)
    selected_question_types.append("difference_description")  # Ensure `difference_description` is always included

    # Generate the conversation as question-answer pairs
    new_conversations = []
    for idx, question_type in enumerate(selected_question_types):
        question = random.choice(question_templates[question_type])
        answer = random.choice(answer_templates[question_type])
        if idx == 0:
            item = [ { "from": "human", "value": f"<image>\n{question}" }, { "from": "gpt", "value": answer } ]
        else:
            item = [ { "from": "human", "value": question }, { "from": "gpt", "value": answer } ]
        new_conversations.extend(item)

    return {"conversations": new_conversations}

def write_sample_to_folder(main_path, folder, sample_images, sample_texts, sample_index):
    """
    Writes sample images and texts to a folder.

    Parameters:
        folder (str): Path to the directory where the samples will be saved.
        sample_images (list): List of PIL images.
        sample_texts (list): List of conversation data.
        sample_index (int): Sample index to name the files.
    """
    image_path = os.path.join(main_path, "images", folder)
    json_path = os.path.join(main_path, "json")
    # Create a new subfolder for each sample
    sample_folder = os.path.join(image_path, f"sample_{sample_index:08d}")
    os.makedirs(sample_folder, exist_ok=True)
    new_conversations = []
    for (img, image_index), conversation in zip(sample_images,sample_texts):
        img_filename = os.path.join(sample_folder, f"image_{image_index}.jpg")
        img.save(img_filename, format="JPEG")
        conversation["image"] = img_filename
        new_conversations.append(conversation)
    os.makedirs(json_path, exist_ok=True)
    # Save the conversation as a JSON file
    json_filename = os.path.join(json_path, f"{folder}.json")
    with open(json_filename, 'w') as f:
        json.dump(new_conversations, f, indent=2)

def generate_dataset(main_path, total_images, images_per_folder, tokenizer_model, max_tokens_per_sample=8192, image_token_length=128):
    folder_count = 0
    image_count = 0
    os.makedirs(main_path, exist_ok=True)
    icon_path = "Fonts/svgs/"
    icon_items = glob(os.path.join(icon_path, "*", "*.svg"))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()

    # Set up multiprocessing
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    # Arguments for generating images
    args = [(random.choice(icon_items), 256, 7, 'white') for _ in range(total_images)]

    # Progress bar for generating images
    with tqdm(total=total_images, desc="Generating Images") as pbar:
        current_sample_images = []
        current_sample_texts = []
        current_sample_token_length = 0
        sample_counter = 0

        for result in pool.imap_unordered(generate_size_image, args):
            img, shape, target_pos, size_ratio, num_objects, color = result
            conversations = generate_conversation_for_icon(shape, target_pos, size_ratio, 7, num_objects)

            # Compute token length of the conversation
            text_token_length = sum(len(tokenizer.encode(message["value"])) for message in conversations["conversations"])
            total_token_length = image_token_length + text_token_length

            # Check if adding this image exceeds the max tokens per sample
            if current_sample_token_length + total_token_length > max_tokens_per_sample:
                # Save current sample to a new folder
                folder_count += 1
                folder_path = f"size_{folder_count:05d}"

                write_sample_to_folder(main_path, folder_path, current_sample_images, current_sample_texts, sample_counter)
                sample_counter += 1
                current_sample_images = []
                current_sample_texts = []
                current_sample_token_length = 0

            # Add image and description to current sample
            current_sample_images.append((img, image_count))
            current_sample_texts.append(conversations)
            current_sample_token_length += total_token_length

            image_count += 1
            pbar.update(1)

            # If images per folder limit is reached, reset
            if sample_counter >= images_per_folder:
                sample_counter = 0

        # Write any remaining samples to a final folder
        if current_sample_images:
            folder_count += 1
            folder_path = f"size_{folder_count:05d}"
            write_sample_to_folder(main_path, folder_path, current_sample_images, current_sample_texts, sample_counter)

    # Close and join the pool to free resources
    pool.close()
    pool.join()

    print(f"Dataset generation complete. Total images: {image_count}")
    print(f"Dataset saved in {main_path}")

if __name__ == "__main__":
    args = parse_arguments()
    args.main_path = os.path.join(args.main_path, "size")
    os.makedirs(args.main_path, exist_ok=True)
    generate_dataset(args.main_path, args.total_images, args.images_per_folder, args.tokenizer_model)

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
from svglib.svglib import svg2rlg
from reportlab.lib.colors import Color
from reportlab.graphics import renderPM
from glob import glob
from transformers import AutoTokenizer
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate orientation image dataset")
    parser.add_argument("--main_path", type=str, default="saliency_data_sft_v2/orientation", help="Path to save dataset")
    parser.add_argument("--total_images", type=int, default=166000, help="Total number of images to generate")
    parser.add_argument("--images_per_folder", type=int, default=1000, help="Number of images per folder")
    parser.add_argument("--tokenizer_model", type=str, default="gpt2", help="Tokenizer model to use")
    return parser.parse_args()

def load_and_color_icon(icon_path, size, color):
    drawing = svg2rlg(icon_path)

    # Apply color to the icon
    def apply_color(shape):
        if hasattr(shape, 'fillColor'):
            shape.fillColor = Color(color[0]/255, color[1]/255, color[2]/255, alpha=1)
        if hasattr(shape, 'strokeColor'):
            shape.strokeColor = Color(color[0]/255, color[1]/255, color[2]/255, color[3]/255)
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
    img = img.resize((size, size), Image.LANCZOS)
    
    return img

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

def conversation_generation_orientation(shape, target_pos, angle_diff, grid_size, num_objects):
    """
    Generate a conversation in the form of question-answer pairs about an image
    based on its orientation characteristics.

    Parameters:
        shape (str): The shape of the objects (e.g., "square", "icon").
        target_pos (tuple): The (x, y) position of the uniquely-oriented object in the grid.
        angle_diff (float): Angle difference of the unique object's orientation.
        grid_size (int): Dimension of the grid (e.g., 7x7).
        num_objects (int): Total number of icons in the grid.

    Returns:
        list of dict: A conversation represented as a list of question-answer pairs.
    """

    # Normalize coordinates of target position
    norm_x = round((target_pos[0] + 0.5) / grid_size, 2)
    norm_y = round((target_pos[1] + 0.5) / grid_size, 2)

    # Question and Answer templates
    question_templates = {
        "object_count": [
            "How many objects are displayed in this image?",
            "Can you tell me the number of shapes present?",
            "What is the total number of icons shown in the grid?"
        ],
        "shape_type": [
            "What shape are these objects?",
            "Are all the objects the same shape, or are there different shapes?",
            "Can you describe the shape of each icon?"
        ],
        "unique_orientation": [
            "Is there any icon that has a different orientation?",
            "Do you notice any icon rotated differently from the others?",
            "Is one of the objects oriented in a unique way?"
        ],
        "unique_orientation_position": [
            "Where is the differently-oriented icon located?",
            "Can you tell me the position of the rotated object?",
            "Where can I find the distinctively oriented object in the grid?"
        ],
        "difference_description": [
            "What makes this unique icon different?",
            "How does this icon stand out from the rest?",
            "What unique feature does this icon have compared to others?"
        ]
    }

    answer_templates = {
        "object_count": [
            f"There are {num_objects} icons displayed in this image.",
            f"The grid contains a total of {num_objects} objects.",
            f"You'll find {num_objects} objects here."
        ],
        "shape_type": [
            f"Each icon is shaped like a {shape}.",
            f"The shapes in this image are all {shape}s.",
            f"All objects in the grid are {shape}-shaped."
        ],
        "unique_orientation": [
            "Yes, one icon has a unique orientation compared to the others.",
            "There is one object that differs in orientation.",
            "One icon stands out due to its different rotation."
        ],
        "unique_orientation_position": [
            f"The distinctively oriented icon is located at normalized coordinates ({norm_x}, {norm_y}).",
            f"You'll find the rotated object around ({norm_x}, {norm_y}).",
            f"The unique icon can be found at ({norm_x}, {norm_y}) in normalized coordinates."
        ],
        "difference_description": [
            f"This unique icon is rotated approximately {angle_diff:.1f} degrees differently from the others.",
            f"The distinguishing feature of this icon is its orientation, rotated {angle_diff:.1f} degrees.",
            f"It stands out by being rotated around {angle_diff:.1f} degrees from the standard orientation."
        ]
    }

    # Select random questions and answers for diversity
    selected_question_types = random.sample(["object_count", "shape_type", "unique_orientation", "unique_orientation_position"], 4)
    selected_question_types.append("difference_description")  # Always include the difference description

    # Generate the conversation as a list of question-answer pairs
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

def generate_orientation_image(args):
    icon_path, size, grid_size, background_color = args
    img = Image.new('RGB', (size, size), background_color)
    
    base_angle = 0
    target_angle = random.uniform(45, 315)
    
    cell_size = size // grid_size
    icon_size = int(cell_size * 0.8)
    
    # Random color for this image
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Load and color the icon
    icon = load_and_color_icon(icon_path, icon_size, color)
    
    # Vary the number of objects
    min_objects = (grid_size * grid_size) // 2
    max_objects = grid_size * grid_size
    num_objects = random.randint(min_objects, max_objects)
    
    # Randomly select positions for icons
    positions = random.sample([(x, y) for x in range(grid_size) for y in range(grid_size)], num_objects)
    
    # Select target position
    target_pos = random.choice(positions)
    
    for x, y in positions:
        angle = target_angle if (x, y) == target_pos else base_angle
        
        rotated_icon = icon.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        paste_x = x * cell_size + (cell_size - icon_size) // 2
        paste_y = y * cell_size + (cell_size - icon_size) // 2
        
        img.paste(rotated_icon, (paste_x, paste_y), rotated_icon)
    
    return img, os.path.basename(icon_path).split('.')[0], target_pos, target_angle, num_objects, color

def generate_dataset(main_path, total_images, images_per_folder, tokenizer_model, max_tokens_per_sample=8192, image_token_length=128):
    folder_count = 0
    image_count = 0
    os.makedirs(main_path, exist_ok=True)
    icon_path = "Fonts/svgs/regular"
    icon_items = glob(os.path.join(icon_path, "*.svg"))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()

    # Set up multiprocessing
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    # Arguments for generating images
    args = [(random.choice(icon_items), 400, 7, 'white') for _ in range(total_images)]

    # Progress bar for generating images
    with tqdm(total=total_images, desc="Generating Images") as pbar:
        current_sample_images = []
        current_sample_texts = []
        current_sample_token_length = 0
        sample_counter = 0

        for result in pool.imap_unordered(generate_orientation_image, args):
            img, shape, target_pos, angle_diff, num_objects, color = result
            conversations = conversation_generation_orientation(shape, target_pos, angle_diff, 7, num_objects)
                                                                
            # Compute token length of the conversation
            text_token_length = sum(len(tokenizer.encode(message["value"])) for message in conversations["conversations"])
            total_token_length = image_token_length + text_token_length

            # Check if adding this image exceeds the max tokens per sample
            if current_sample_token_length + total_token_length > max_tokens_per_sample:
                # Save current sample to a new folder
                folder_count += 1
                folder_path = f"orientation_{folder_count:05d}"

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
            folder_path = f"orientation_{folder_count:05d}"
            write_sample_to_folder(main_path, folder_path, current_sample_images, current_sample_texts, sample_counter)

    # Close and join the pool to free resources
    pool.close()
    pool.join()

    print(f"Dataset generation complete. Total images: {image_count}")
    print(f"Dataset saved in {main_path}")

if __name__ == "__main__":
    args = parse_arguments()
    args.main_path = os.path.join(args.main_path, "orientation")
    os.makedirs(args.main_path, exist_ok=True)
    generate_dataset(args.main_path, args.total_images, args.images_per_folder, args.tokenizer_model)

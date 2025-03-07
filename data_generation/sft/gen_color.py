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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate psychophysical image dataset")
    parser.add_argument("--main_path", type=str, default="saliency_data_sft_v2/color", help="Path to save dataset")
    parser.add_argument("--total_images", type=int, default=166000, help="Total number of images to generate")
    parser.add_argument("--images_per_folder", type=int, default=1000, help="Number of images per folder")
    parser.add_argument("--tokenizer_model", type=str, default="gpt2", help="Tokenizer model to use")
    return parser.parse_args()


def create_shape(shape, size, color, angle=0):
    scale = 4
    img_size = size * 2 * scale
    img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    center = img_size // 2
    shape_size = size * scale
    
    if shape == 'circle':
        draw.ellipse([center - shape_size, center - shape_size, 
                      center + shape_size, center + shape_size], fill=color)
    elif shape == 'square':
        draw.rectangle([center - shape_size, center - shape_size, 
                        center + shape_size, center + shape_size], fill=color)
    elif shape == 'triangle':
        points = [(center, center - shape_size),
                  (center - shape_size * math.sin(math.pi/3), center + shape_size * math.cos(math.pi/3)),
                  (center + shape_size * math.sin(math.pi/3), center + shape_size * math.cos(math.pi/3))]
        draw.polygon(points, fill=color)
    elif shape == 'rectangle':
        draw.rectangle([center - shape_size, center - shape_size/2, 
                        center + shape_size, center + shape_size/2], fill=color)
    elif shape == 'ellipse':
        draw.ellipse([center - shape_size, center - shape_size/2, 
                      center + shape_size, center + shape_size/2], fill=color)
    elif shape in ['pentagon', 'hexagon', 'star']:
        points = []
        n_points = 5 if shape == 'pentagon' else 6 if shape == 'hexagon' else 10
        for i in range(n_points):
            angle_pt = i * (2 * math.pi / n_points) - math.pi/2
            r = shape_size if shape != 'star' or i % 2 == 0 else shape_size / 2
            x = center + r * math.cos(angle_pt)
            y = center + r * math.sin(angle_pt)
            points.append((x, y))
        draw.polygon(points, fill=color)
    elif shape == 'diamond':
        draw.polygon([(center, center - shape_size), (center + shape_size, center),
                      (center, center + shape_size), (center - shape_size, center)], fill=color)
    elif shape == 'cross':
        thickness = shape_size // 3
        draw.rectangle([center - thickness, center - shape_size, 
                        center + thickness, center + shape_size], fill=color)
        draw.rectangle([center - shape_size, center - thickness, 
                        center + shape_size, center + thickness], fill=color)
    
    img = img.rotate(angle, resample=Image.NEAREST, expand=False)
    img = img.resize((size * 2, size * 2), Image.NEAREST)
    return img


def generate_psychophysical_image(args):
    size, num_shapes, background_color = args
    img = Image.new('RGB', (size, size), background_color)
    shapes = ['circle', 'square', 'triangle', 'rectangle', 'ellipse', 'pentagon', 'hexagon', 'star', 'diamond', 'cross']
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
    num_shapes = random.randint(15, num_shapes)
    chosen_shapes = random.sample(shapes, 2)
    target_color, distractor_color = random.sample(colors, 2)
    
    placed_shapes = []
    target_pos = None
    
    for i in range(num_shapes):
        shape = chosen_shapes[i % 2]
        color = target_color if i == 0 else distractor_color
        shape_size = random.randint(size // 30, size // 15)
        angle = random.randint(0, 359)
        
        attempts = 0
        while attempts < 100:
            x = random.randint(shape_size, size - shape_size)
            y = random.randint(shape_size, size - shape_size)
            
            if all((x-ox)**2 + (y-oy)**2 > (shape_size + os + 5)**2 for ox, oy, os in placed_shapes):
                shape_img = create_shape(shape, shape_size, ImageColor.getrgb(color), angle)
                img.paste(shape_img, (x - shape_size, y - shape_size), shape_img)
                placed_shapes.append((x, y, shape_size))
                if i == 0:
                    target_pos = (x, y)
                break
            
            attempts += 1
    
    return img, chosen_shapes, chosen_shapes[0], target_color, distractor_color, target_pos

def conversation_generation(chosen_shapes, target_shape, target_color, distractor_color, target_pos, size):
    """
    Generate a conversation in the form of question-answer pairs about an image
    based on its attributes.

    Parameters:
        chosen_shapes (list): List of shape types used in the image (e.g., ["circle", "square"]).
        target_shape (str): The unique shape that differs in some characteristic (size, color, etc.).
        target_color (str): The color of the target shape.
        distractor_color (str): The color of other shapes.
        target_pos (tuple): The (x, y) position of the unique shape in the image.
        size (int): Size of the image for normalizing coordinates.

    Returns:
        list of dict: A conversation represented as a list of question-answer pairs.
    """
    # Normalize coordinates of target position
    norm_x = round(target_pos[0] / size, 2)
    norm_y = round(target_pos[1] / size, 2)

    # Shape descriptions for questions and answers
    shape_descriptions = {
        'circle': 'circular object',
        'square': 'square-shaped object',
        'triangle': 'triangular object',
        'rectangle': 'rectangular object',
        'ellipse': 'oval-shaped object',
        'pentagon': 'pentagonal object',
        'hexagon': 'hexagonal object',
        'star': 'star-shaped object',
        'diamond': 'diamond-shaped object',
        'cross': 'cross-shaped object'
    }

    # Target and distractor descriptions
    target_desc = shape_descriptions[target_shape]
    distractor_desc = shape_descriptions[chosen_shapes[1] if chosen_shapes[0] == target_shape else chosen_shapes[0]]

    # Question and Answer templates
    question_templates = {
        "object_count": [
            "How many objects are there in the image?",
            "What is the total number of shapes present?",
            "Can you tell me the number of items displayed in the image?"
        ],
        "shape_type": [
            "What shape are the objects in this image?",
            "Can you describe the shape of each object?",
            "Are all the objects the same shape, or are there different shapes?"
        ],
        "unique_shape": [
            "Is there any object with a different characteristic?",
            "Do you notice any object that stands out?",
            "Is one of the objects different from the others?"
        ],
        "unique_shape_position": [
            "Where is the unique object located?",
            "Can you tell me the position of the different object?",
            "Where can I find the distinct object in the image?"
        ],
        "difference_description": [
            "What makes the unique object different?",
            "How does this object stand out from the rest?",
            "What is special about this object compared to the others?"
        ]
    }

    # Answer templates
    answer_templates = {
        "object_count": [
            f"There are several objects displayed, specifically {size // 10}.",
            f"You'll find around {size // 10} items in the image.",
            f"This image contains approximately {size // 10} objects."
        ],
        "shape_type": [
            f"The shapes in the image are mainly {target_desc} and {distractor_desc}.",
            f"Most objects are {distractor_desc}, while one is a {target_desc}.",
            f"The image shows both {target_desc}s and {distractor_desc}s."
        ],
        "unique_shape": [
            "Yes, one object has a unique characteristic compared to the others.",
            "There is indeed one object that differs.",
            "One object stands out due to a unique attribute."
        ],
        "unique_shape_position": [
            f"The unique object is located at normalized coordinates ({norm_x}, {norm_y}).",
            f"You'll find the distinct object around the position ({norm_x}, {norm_y}).",
            f"The unique object can be pinpointed at ({norm_x}, {norm_y}) in normalized coordinates."
        ],
        "difference_description": [
            f"This unique object is colored {target_color}, unlike the others which are {distractor_color}.",
            f"The primary difference is its color: it is {target_color} while others are {distractor_color}.",
            f"One object is {target_color} while the rest are {distractor_color}, making it stand out."
        ]
    }

    # Select random questions and answers for diversity
    selected_question_types = random.sample(["object_count", "shape_type", "unique_shape", "unique_shape_position"], 4)
    selected_question_types.append("difference_description")  # Always include difference description

    # Generate conversation as a list of question-answer pairs
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
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()

    # Set up multiprocessing
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    # Arguments for generating images
    background_colors = ['white', 'lightgray', 'lightblue', 'lightyellow', 'lightgreen']
    args = [(400, 40, random.choice(background_colors)) for _ in range(total_images)]

    # Progress bar for generating images
    with tqdm(total=total_images, desc="Generating Images") as pbar:
        current_sample_images = []
        current_sample_texts = []
        current_sample_token_length = 0
        sample_counter = 0

        for result in pool.imap_unordered(generate_psychophysical_image, args):
            img, chosen_shapes, target_shape, target_color, distractor_color, target_pos = result
            conversations = conversation_generation(chosen_shapes, target_shape, target_color, distractor_color, target_pos, 400)

            # Compute token length of the conversation
            text_token_length = sum(len(tokenizer.encode(message["value"])) for message in conversations["conversations"])
            total_token_length = image_token_length + text_token_length

            # Check if adding this image exceeds the max tokens per sample
            if current_sample_token_length + total_token_length > max_tokens_per_sample:
                # Save current sample to a new folder
                folder_count += 1
                folder_path = f"color_{folder_count:05d}"

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
            folder_path = f"color_{folder_count:05d}"
            write_sample_to_folder(main_path, folder_path, current_sample_images, current_sample_texts, sample_counter)
    # Close and join the pool to free resources
    pool.close()
    pool.join()

    print(f"Dataset generation complete. Total images: {image_count}")
    print(f"Dataset saved in {main_path}")

if __name__ == "__main__":
    args = parse_arguments()
    args.main_path = os.path.join(args.main_path, "color")
    os.makedirs(args.main_path, exist_ok=True)
    generate_dataset(args.main_path, args.total_images, args.images_per_folder, args.tokenizer_model)

import argparse
import random
import math
import multiprocessing
from PIL import Image
import io
import json
import tarfile
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from reportlab.lib.colors import Color
from tqdm import tqdm
from glob import glob
from transformers import AutoTokenizer
import tempfile
import shutil
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate size variation image dataset")
    parser.add_argument("--main_path", type=str, default="/lustre1/tier2/projects/falcon-mm/data/p3o3/pretrain", help="Path to save dataset")
    parser.add_argument("--total_images", type=int, default=333000, help="Total number of images to generate")
    parser.add_argument("--images_per_tar", type=int, default=1000, help="Number of images per tar file")
    parser.add_argument("--tokenizer_model", type=str, default="gpt2", help="Tokenizer model to use")
    return parser.parse_args()



def get_size_description(size_ratio):
    if size_ratio < 1.0:
        if size_ratio >= 0.9:
            return "slightly smaller"
        elif size_ratio >= 0.8:
            return "smaller"
        else:
            return "much smaller"
    elif size_ratio == 1.0:
        return "the same size"
    else:
        if size_ratio <= 1.2:
            return "slightly bigger"
        elif size_ratio <= 1.5:
            return "bigger"
        else:
            return "much bigger"

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

def generate_image_description(shape, target_pos, size_ratio, grid_size, num_objects, color):
    norm_x = round((target_pos[0] + 0.5) / grid_size, 2)
    norm_y = round((target_pos[1] + 0.5) / grid_size, 2)

    size_description = get_size_description(size_ratio)

    prompts = [
        f"This image contains {num_objects} {shape} icons. "
        f"All icons are of the same size except for one. "
        f"The {shape} with a different size is located at the normalized coordinates ({norm_x}, {norm_y}). "
        f"It is {size_description} than the others.",

        f"A collection of {num_objects} {shape} icons is displayed in this image. "
        f"Among them, a single {shape} stands out with a unique size. "
        f"This distinct icon can be found at the position ({norm_x}, {norm_y}) in normalized coordinates. "
        f"It is {size_description} compared to its counterparts.",

        f"The image showcases {num_objects} {shape} icons. "
        f"While most icons share a common size, one breaks the pattern. "
        f"Located at ({norm_x}, {norm_y}) in the normalized coordinate system, "
        f"this outlier {shape} is {size_description} than the rest.",

        f"An array of {num_objects} {shape} icons is presented. "
        f"Amidst the uniformity, a single {shape} icon stands out due to its unique size. "
        f"This distinctive element can be pinpointed at the normalized coordinates ({norm_x}, {norm_y}), "
        f"being {size_description} than its peers.",

        f"This visual composition features {num_objects} {shape} icons. "
        f"A solitary {shape} disrupts the size harmony of the group. "
        f"This unique element is positioned at ({norm_x}, {norm_y}) in normalized space, "
        f"being {size_description} than the norm."
    ]

    return random.choice(prompts)

def write_sample_to_tar(tar, sample_images, sample_texts, sample_index, temp_dir):
    # Prepare texts
    texts_str = ''.join([f'<image>{caption}<|endoftext|>' for caption in sample_texts])

    # Convert images to a list of PIL images and save as numpy array
    images_pil_list = [img for img, _ in sample_images]
    npy_filename = os.path.join(temp_dir, f"sample_{sample_index:08d}_images.npy")
    with open(npy_filename, 'wb') as f:
        np.save(f, images_pil_list, allow_pickle=True)

    # Now write to tar file
    sample_name = f'sample_{sample_index:08d}'

    # Add npy file to tar
    with open(npy_filename, 'rb') as f:
        npy_bytes = f.read()
    npy_info = tarfile.TarInfo(name=f'{sample_name}.npy')
    npy_info.size = len(npy_bytes)
    tar.addfile(npy_info, io.BytesIO(npy_bytes))

    # Write texts txt file
    text_bytes = texts_str.encode('utf-8')
    txt_info = tarfile.TarInfo(name=sample_name + '.txt')
    txt_info.size = len(text_bytes)
    tar.addfile(txt_info, io.BytesIO(text_bytes))

    # Remove temporary numpy file
    os.remove(npy_filename)
    
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

def generate_dataset(main_path, total_images, images_per_tar, tokenizer_model, max_tokens_per_sample=2048, image_token_length=576):
    tar_count = 0
    image_count = 0
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
    args = [(random.choice(icon_items), 400, 7, 'white') for _ in range(total_images)]

    # Progress bar for generating images
    with tqdm(total=total_images, desc="Generating Images") as pbar:
        current_sample_images = []
        current_sample_texts = []
        current_sample_token_length = 0
        sample_counter = 0

        tar = None

        for result in pool.imap_unordered(generate_size_image, args):
            img, shape, target_pos, size_ratio, num_objects, color = result
            description = generate_image_description(shape, target_pos, size_ratio, 7, num_objects, color)

            # Compute token length of the description
            text_token_length = len(tokenizer.encode(description))
            total_token_length = image_token_length + text_token_length

            # Check if adding this image exceeds the max tokens per sample
            if current_sample_token_length + total_token_length > max_tokens_per_sample:
                # Write current sample to tar
                if tar is None:
                    tar_count += 1
                    tar_filename = f"size_variation_{tar_count:05d}.tar"
                    tar_filename = os.path.join(main_path, tar_filename)
                    tar = tarfile.open(tar_filename, "w")

                write_sample_to_tar(tar, current_sample_images, current_sample_texts, sample_counter, temp_dir)
                sample_counter += 1
                current_sample_images = []
                current_sample_texts = []
                current_sample_token_length = 0

            # Add image and description to current sample
            current_sample_images.append((img, image_count))
            current_sample_texts.append(description)
            current_sample_token_length += total_token_length

            image_count += 1
            pbar.update(1)

            # Write tar file if images_per_tar samples are reached
            if sample_counter >= images_per_tar:
                if tar is not None:
                    tar.close()
                tar = None
                sample_counter = 0

        # Write any remaining samples to tar
        if current_sample_images:
            if tar is None:
                tar_count += 1
                tar_filename = f"size_variation_{tar_count:05d}.tar"
                tar_filename = os.path.join(main_path, tar_filename)
                tar = tarfile.open(tar_filename, "w")

            write_sample_to_tar(tar, current_sample_images, current_sample_texts, sample_counter, temp_dir)

        if tar is not None:
            tar.close()

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    print(f"Dataset generation complete. Total images: {image_count}")
    print(f"Dataset saved in {main_path}")

if __name__ == "__main__":
    args = parse_arguments()
    args.main_path = os.path.join(args.main_path, "size")
    os.makedirs(args.main_path, exist_ok=True)
    generate_dataset(args.main_path, args.total_images, args.images_per_tar, args.tokenizer_model)

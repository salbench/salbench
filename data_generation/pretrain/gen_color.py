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
    parser.add_argument("--main_path", type=str, default="./data/pretrain/", help="Path to save dataset")
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

def generate_image_description(chosen_shapes, target_shape, target_color, distractor_color, target_pos, size):
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

    target_desc = shape_descriptions[target_shape]
    distractor_desc = shape_descriptions[chosen_shapes[1] if chosen_shapes[0] == target_shape else chosen_shapes[0]]

    # Normalize coordinates to be between 0 and 1
    norm_x = round(target_pos[0] / size, 2)
    norm_y = round(target_pos[1] / size, 2)

    descriptions = [
        f"This image contains a salient {target_desc} colored {target_color} among numerous {distractor_desc}s colored {distractor_color}. The distinct {target_desc} is located at the normalized coordinates ({norm_x}, {norm_y}).",
        f"A {target_color} {target_desc} stands out from a collection of {distractor_color} {distractor_desc}s in this image. The unique {target_desc} can be found at the relative position ({norm_x}, {norm_y}).",
        f"The visual scene presents a {target_color} {target_desc} as the target, surrounded by multiple {distractor_color} {distractor_desc}s serving as distractors. The target {target_desc} is positioned at ({norm_x}, {norm_y}) in normalized image coordinates.",
        f"Amidst a group of {distractor_color} {distractor_desc}s, a single {target_color} {target_desc} captures attention. This distinctive element is situated at ({norm_x}, {norm_y}) when the image is viewed on a 0 to 1 scale.",
        f"The image features a conspicuous {target_color} {target_desc}, contrasting with several {distractor_color} {distractor_desc}s. The focal {target_desc} is located at ({norm_x}, {norm_y}) in the normalized coordinate system of the image."
    ]

    return random.choice(descriptions)

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


def generate_dataset(main_path, total_images, images_per_tar, tokenizer_model, max_tokens_per_sample=2048, image_token_length=576):
    tar_count = 0
    image_count = 0
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()

    # Set up multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 4)
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

        tar = None

        for result in pool.imap_unordered(generate_psychophysical_image, args):
            img, chosen_shapes, target_shape, target_color, distractor_color, target_pos = result
            description = generate_image_description(chosen_shapes, target_shape, target_color, distractor_color, target_pos, 308)

            # Compute token length of the description
            text_token_length = len(tokenizer.encode(description))
            total_token_length = image_token_length + text_token_length

            # Check if adding this image exceeds the max tokens per sample
            if current_sample_token_length + total_token_length > max_tokens_per_sample:
                # Write current sample to tar
                if tar is None:
                    tar_count += 1
                    tar_filename = f"color_{tar_count:05d}.tar"
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
                tar_filename = f"color_{tar_count:05d}.tar"
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
    args.main_path = os.path.join(args.main_path, "color")
    os.makedirs(args.main_path, exist_ok=True)
    generate_dataset(args.main_path, args.total_images, args.images_per_tar, args.tokenizer_model)

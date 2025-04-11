import argparse
import glob
from PIL import Image
from collections import deque
import numpy as np
import logging
from tqdm.asyncio import tqdm_asyncio
import asyncio

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Concatenate images based on overlapping pixels.')
    parser.add_argument('glob_pattern', type=str, nargs='+', help='Glob pattern(s) to match image files.')
    parser.add_argument('-o', '--orientation', choices=['vertical', 'horizontal'], default='vertical',
                        help='Orientation to concatenate images (default: vertical).')
    parser.add_argument('-r', '--result', default='./pilstack.png', help='Output file path (default: ./pilstack.png)')
    parser.add_argument('--remove-duplicates', action='store_true', default=True,
                        help='Remove all but the last occurrence of duplicated regions (default: True)')
    parser.add_argument('--keep-duplicates', dest='remove_duplicates', action='store_false',
                        help='Keep all duplicated regions.')
    return parser.parse_args()

async def open_image_async(file):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, Image.open, file)

async def open_images(glob_patterns):
    files = []
    for pattern in glob_patterns:
        files.extend(glob.glob(pattern))

    if len(files) < 2:
        raise ValueError("At least two input files are required.")

    images = []
    for file in tqdm_asyncio(files, desc="Opening images"):
        try:
            img = await open_image_async(file)
            images.append((file, img.convert('RGBA')))
        except Exception as e:
            logging.warning(f"Could not open {file}: {e}")

    if len(images) < 2:
        raise ValueError("At least two valid images must be opened.")

    return images


def find_overlap(im1, im2, orientation):
    arr1 = np.array(im1)
    arr2 = np.array(im2)

    if orientation == 'vertical':
        max_overlap = min(arr1.shape[0], arr2.shape[0])
        for overlap in range(max_overlap, 0, -1):
            if np.array_equal(arr1[-overlap:], arr2[:overlap]):
                return overlap
    else:
        max_overlap = min(arr1.shape[1], arr2.shape[1])
        for overlap in range(max_overlap, 0, -1):
            if np.array_equal(arr1[:, -overlap:], arr2[:, :overlap]):
                return overlap

    return 0


def identify_repeated_regions(images, orientation):
    repeated_regions = set()
    min_size = min(img.size[1 if orientation == 'vertical' else 0] for _, img in images)
    threshold_size = min_size // 2

    region_counts = {}

    for _, img in images:
        arr = np.array(img)
        key = arr[:, :threshold_size].tobytes() if orientation == 'vertical' else arr[:threshold_size, :].tobytes()
        region_counts[key] = region_counts.get(key, 0) + 1

    for region, count in region_counts.items():
        if count >= len(images) - 1:
            repeated_regions.add(region)

    return repeated_regions


def remove_duplicate_regions(images, orientation):
    repeated_regions = identify_repeated_regions(images, orientation)
    unique_images = []

    seen_regions = set()
    for file, img in reversed(images):
        arr = np.array(img)
        region = arr.tobytes()
        if region not in seen_regions or region not in repeated_regions:
            unique_images.insert(0, (file, img))
            seen_regions.add(region)

    return unique_images

async def concat_images(images, orientation, remove_duplicates):
    if remove_duplicates:
        images = remove_duplicate_regions(images, orientation)

    img_queue = deque(images)
    base_name, base_img = img_queue.popleft()
    positions = [(base_name, (0, 0))]

    progress_bar = tqdm_asyncio(total=len(images)-1, desc="Concatenating images")

    while img_queue:
        matched = False
        for i in range(len(img_queue)):
            name, next_img = img_queue[i]
            overlap = find_overlap(base_img, next_img, orientation)
            if overlap > 0:
                if orientation == 'vertical':
                    new_img = Image.new('RGBA', (max(base_img.width, next_img.width),
                                                 base_img.height + next_img.height - overlap))
                    new_img.paste(base_img, (0, 0))
                    new_img.paste(next_img, (0, base_img.height - overlap))
                else:
                    new_img = Image.new('RGBA', (base_img.width + next_img.width - overlap,
                                                 max(base_img.height, next_img.height)))
                    new_img.paste(base_img, (0, 0))
                    new_img.paste(next_img, (base_img.width - overlap, 0))

                positions.append((name, (0, base_img.height - overlap) if orientation == 'vertical' else (base_img.width - overlap, 0)))
                base_img = new_img
                img_queue.remove((name, next_img))
                matched = True
                progress_bar.update(1)
                break

        if not matched:
            progress_bar.close()
            raise ValueError("Could not find sufficient overlap between remaining images.")

    progress_bar.close()
    return base_img, positions

async def main():
    args = parse_args()
    images = await open_images(args.glob_pattern)

    result_img, positions = await concat_images(images, args.orientation, args.remove_duplicates)

    logging.info("Concatenation successful. Image positions:")
    for name, pos in positions:
        logging.info(f"{name}: {pos}")

    result_img.save(args.result)
    logging.info(f"Saved {args.result}")


if __name__ == '__main__':
    asyncio.run(main())

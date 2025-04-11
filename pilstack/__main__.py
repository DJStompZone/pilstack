import argparse
import glob
from PIL import Image
from collections import deque
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Concatenate images based on overlapping pixels.')
    parser.add_argument('glob_pattern', type=str, help='Glob pattern to match image files.')
    parser.add_argument('-o', '--orientation', choices=['vertical', 'horizontal'], default='vertical',
                        help='Orientation to concatenate images (default: vertical).')
    return parser.parse_args()


def open_images(glob_pattern):
    files = glob.glob(glob_pattern)
    if len(files) < 2:
        raise ValueError("At least two input files are required.")

    images = []
    for file in files:
        try:
            img = Image.open(file).convert('RGBA')
            images.append((file, img))
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


def concat_images(images, orientation):
    img_queue = deque(images)
    base_name, base_img = img_queue.popleft()
    positions = [(base_name, (0, 0))]

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
                break

        if not matched:
            raise ValueError("Could not find sufficient overlap between remaining images.")

    return base_img, positions


def main():
    args = parse_args()
    images = open_images(args.glob_pattern)

    result_img, positions = concat_images(images, args.orientation)

    logging.info("Concatenation successful. Image positions:")
    for name, pos in positions:
        logging.info(f"{name}: {pos}")

    result_img.save('concatenated_image.png')
    logging.info("Saved concatenated_image.png")


if __name__ == '__main__':
    main()

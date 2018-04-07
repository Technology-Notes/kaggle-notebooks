# -*- coding: utf-8 -*-

# !/usr/bin/python

"""
img_prepare.py

Does the following:

  1. Take images downloaded using `img_downloader.py`.
  2. Save into a folder-per-class directory structure.
  3. Rescale the image to match target dimensions.
"""

import random
import os
import argparse
import math
from tqdm import tqdm
import multiprocessing

import pandas as pd
from PIL import Image


def scale_to(x, ratio, targ):
    return max(math.floor(x * ratio), targ)


def resize_img(filename, target_size, path, new_path):
    """Resize an image and save to a new path."""
    dest = os.path.join(new_path, filename)
    if os.path.exists(dest):
        return

    try:
        im = Image.open(os.path.join(path, filename)).convert('RGB')
    except (OSError):
        return

    r, c = im.size
    ratio = target_size / min(r, c)
    new_size = (
        scale_to(r, ratio, target_size),
        scale_to(c, ratio, target_size))
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    im.resize(new_size, Image.LINEAR).save(dest)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-dir', type=str, help='Input directory to read images from.', required=True, dest='input_dir')
parser.add_argument('--input-csv', type=str, help='Path to CSV with class info.', required=True, dest='input_csv')
parser.add_argument('--output-dir', help='Output directory to save files to', type=str, dest='output_dir')
parser.add_argument('--size', help='Size to rescale image', type=int, default=64)
parser.add_argument('--val-ratio', help='Ratio of validation set', type=float, default=0.2, dest='val_ratio')
parser.add_argument('--seed', help='Seed for generating random numbers', type=int, default=42, dest='seed')

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the CSV
    df = pd.read_csv(args.input_csv)
    iter_df = zip(df.landmark_id.values, df.id.values)

    random.seed(args.seed)

    def _load_and_resize(row):
        landmark_id, img_id = row

        filename = f'{img_id}.jpg'
        # Check if image exists
        dest = os.path.join(args.input_dir, filename)
        if not os.path.exists(dest):
            return

        rand = random.random()

        set_name = 'train'
        if (1 - rand) < args.val_ratio:
            set_name = 'val'

        output_dir = os.path.join(
            args.output_dir, str(args.size), set_name, str(landmark_id))

        # Resize image and add to path.
        resize_img(filename, args.size, args.input_dir, output_dir)

    pool = multiprocessing.Pool(processes=20)  # Num of CPUs
    list(tqdm(pool.imap_unordered(_load_and_resize, iter_df), total=len(df)))
    pool.close()
    pool.terminate()

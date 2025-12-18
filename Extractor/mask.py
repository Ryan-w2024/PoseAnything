import cv2
from parser import read_png_as_pil
from masker import MaskGenerator
from tqdm import tqdm
import csv

# BlumNet
import os
import numpy as np

np.bool8 = np.bool_
np.float_ = np.float64
np.obj2sctype = lambda obj: np.dtype(obj).type

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="first frame")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir")
    parser.add_argument("--skeleton_dir", type=str, required=True, help="output dir")
    parser.add_argument("--metadata", type=str, required=True, help="output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    file_names = []

    with open(args.metadata, mode="r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            full_name = row["file_name"]
            name_without_ext = os.path.splitext(full_name)[0]
            file_names.append(name_without_ext)

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"{args.input_dir} does not exist")
    file_list = [f"{i}.png" for i in file_names]
    if not file_list:
        print(f"warning：{args.input_dir} contains does not contain .png file")
        exit(0)

    processed_count = 0

    mask_generator = MaskGenerator()

    count = 0

    for filename in tqdm(file_list, desc="process", unit="None"):
        try:
            if filename.endswith(".png"):

                video_name = filename[:-len(".png")]
                source_path = os.path.join(args.input_dir, filename)
                output_name = f"{video_name}_id.png"
                output_path = os.path.join(args.output_dir, output_name)

                # check if source_path and skeleton_path have the same shape
                skeleton_path = os.path.join(args.skeleton_dir, video_name, "000.png")
                if os.path.exists(source_path) and os.path.exists(skeleton_path):
                    source_img = cv2.imread(source_path)
                    skeleton_img = cv2.imread(skeleton_path)
                    if source_img is not None and skeleton_img is not None:
                        if source_img.shape != skeleton_img.shape:
                            count = count + 1
                            print(f"image and skeleton have different shape: {filename}")

                first_frame = read_png_as_pil(source_path)

                if os.path.exists(output_path):
                    if os.path.exists(skeleton_path):
                        output_img = cv2.imread(output_path)
                        skeleton_img_check = cv2.imread(skeleton_path)
                        if output_img is not None and skeleton_img_check is not None:
                            if output_img.shape == skeleton_img_check.shape:
                                print(f"file {output_path} exist, skip")
                                continue
                    print(f"file {output_path} exit，skip")
                    continue
                print(output_path)

                skeleton_img = cv2.imread(str(skeleton_path))
                if skeleton_img is None:
                    raise FileNotFoundError(f"cannot read: {skeleton_path}")

                gray = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                white_pixels = np.where(binary == 255)
                if len(white_pixels[0]) == 0:
                    print(f"skeleton image {skeleton_path} has no skeleton")
                    continue

                y_coords, x_coords = white_pixels
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                ref_box = [x_min, y_min, x_max, y_max]
                ref_label = None

                ref_box, ref_label = mask_generator.object_detection_and_segmentation(
                    image_input=first_frame,
                    output_path=output_path,
                    ref_box=ref_box,
                    ref_label=ref_label
                )

        except Exception as e:
            print(f"handle {filename} error: {str(e)}")
            continue


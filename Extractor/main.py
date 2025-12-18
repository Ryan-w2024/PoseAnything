from pathlib import Path
from parser import video_to_images
from masker import MaskGenerator
from skeleton import Skeletonizer
from tqdm import tqdm
import shutil

#libraries supporting BlumNet
import os
import numpy as np
np.bool8 = np.bool_
np.float_ = np.float64
np.obj2sctype = lambda obj: np.dtype(obj).type
from BlumNet.detection.gcd.args_parser import get_args_parser
import pandas as pd


"""
The output folder will be:

./result_dir
|- train
    |- video
    |- skeleton_image
    |- first_frame
    |- metadata.csv
|- tmp
    |- masked_frames
    |- frames
    
"""

if __name__ == "__main__":

    # 创建 ArgumentParser 对象
    parser = get_args_parser()
    parser.add_argument("--result_dir", type=str, required=True, help="base output path")
    parser.add_argument("--csv_path", type=str, required=True, help="path to the input CSV file containing video paths and captions")
    args = parser.parse_args()
    video_folder = f"{args.result_dir}/train/video"  # video folder
    print(video_folder)
    frames_dir = f"{args.result_dir}/tmp/frames"  # frame folder
    print(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    masked_frames_dir = f"{args.result_dir}/tmp/masked_frames/"  # mask
    print(masked_frames_dir)
    os.makedirs(masked_frames_dir, exist_ok=True)
    skeleton_dir = f"{args.result_dir}/train/skeleton_image/"  # skeleton
    os.makedirs(skeleton_dir, exist_ok=True)
    meta_fpath = f"metadata.csv" #metadata

    root = Path(video_folder)
    frame_interval = 1

    mask_generator = MaskGenerator()
    args.num_feature_levels = 3
    args.aux_loss = True
    args.gid = True
    args.out_pts = 128
    if not hasattr(args, 'visual_type'):
        args.visual_type = "branches"
    args.resume = "./BlumNet/exps/demo/checkpoint.pth"
    args.num_feature_levels = 3
    args.aux_loss = True
    args.gid = True
    args.out_pts = 128
    skeletonizer = Skeletonizer(args)

    # Retrieve video metadata
    df = pd.read_csv(args.csv_path)
    video_records = df.to_dict('records')

    for row in tqdm(video_records, desc="video process", unit="video"):
        raw_mp4_path = row.get('video_path')

        if not raw_mp4_path or pd.isna(raw_mp4_path):
            continue

        file_name = Path(raw_mp4_path).stem
        caption_data = row.get('caption')

        if pd.isna(caption_data):
            print(f"Warning: No caption detected for video {raw_mp4_path}. Defaulting to 'A video'.")
            short_caption = "A video"
        else:
            short_caption = str(caption_data)

        # step1: Parse video into frames
        frame_list = video_to_images(raw_mp4_path, frames_dir, frame_interval)

        masked_output_dir = os.path.join(masked_frames_dir, file_name)
        os.makedirs(masked_output_dir, exist_ok=True)
        ref_box = None
        ref_label = None
        condition_image = []
        condition_index = []

        # step2: Mask object
        for i in range(len(frame_list)):
            frame = frame_list[i]
            output_path = os.path.join(masked_output_dir, f"{i:03d}.png")

            ref_box, ref_label = mask_generator.object_detection_and_segmentation(
                image_input=frame,
                output_path=output_path,
                ref_box=ref_box,
                ref_label=ref_label
            )

            if i == 0:
                if ref_box is None:
                    tqdm.write(f"Warning: No mask detected in frame {i}.skip this video")
                    break
                w,h = frame.size
                x1, y1, x2, y2 = ref_box
                if (x2-x1)*(y2-y1)*64 < w*h:
                    tqdm.write(f"warning: Detected object is too small.")

            if ref_box is not None and ref_label is not None:
                condition_image.append(f"{i:03d}.png")
                condition_index.append(i)

        df = pd.DataFrame({
            "file_name": f"{file_name}.mp4",
            "condition_images": [condition_image],
            "text": short_caption,
            "condition_index": [condition_index]
        })

        write_header = not os.path.exists(f"{args.result_dir}/{meta_fpath}") or os.path.getsize(f"{args.result_dir}/{meta_fpath}") == 0
        df.to_csv(f"{args.result_dir}/{meta_fpath}", mode='a', header=write_header, index=False)

        print("Mask done")

        # step3: Extract skeleton
        skeletonizer.skeletonization(masked_output_dir, skeleton_dir)
        print(f"skeleton done, result save to: {skeleton_dir}")

        # step4: Copy necessary files from the tmp folder
        src_first_frame = os.path.join(frames_dir, file_name, "000.png")
        src_masked_frame = os.path.join(masked_output_dir, "000.png")

        dest_first_frame_dir = os.path.join(args.result_dir, "train", "first_frame")
        dest_video_dir = os.path.join(args.result_dir, "train", "video")

        os.makedirs(dest_first_frame_dir, exist_ok=True)
        os.makedirs(dest_video_dir, exist_ok=True)

        dest_first_frame_path = os.path.join(dest_first_frame_dir, f"{file_name}.png")
        dest_masked_id_path = os.path.join(dest_video_dir, f"{file_name}_id.png")

        try:
            if os.path.exists(src_first_frame):
                shutil.copy2(src_first_frame, dest_first_frame_path)
            else:
                tqdm.write(f"Warning: Source first frame not found at {src_first_frame}")

            if os.path.exists(src_masked_frame):
                shutil.copy2(src_masked_frame, dest_masked_id_path)
            else:
                tqdm.write(f"Warning: Source masked frame not found at {src_masked_frame}")

        except Exception as e:
            tqdm.write(f"Error copying files for {file_name}: {e}")



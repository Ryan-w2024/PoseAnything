import os
import cv2
import csv
import ast
import numpy as np
import torch
import torchvision
from PIL import Image
from collections import deque
from skimage.morphology import skeletonize
from einops import repeat
from torch.utils.data import Dataset


# -------------------------- 图像处理工具函数 -------------------------

def crop_and_resize(image, max_size):
    width, height = image.size
    max_dim = max(width, height)
    new_h = int(height * max_size / max_dim) // 32 * 32
    new_w = int(width * max_size / max_dim) // 32 * 32

    image = torchvision.transforms.functional.resize(
        image,
        (new_h, new_w),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    return image


def resize_image(image):
    height, width = image.shape[:2]
    max_size = 832
    max_side = max(height, width)
    final_size = (int(width * max_size / max_side) // 32 * 32, int(height * max_size / max_side) // 32 * 32)
    resized_image = cv2.resize(image, final_size, interpolation=cv2.INTER_LINEAR)
    return resized_image


def load_condition_image(condition_image_paths, condition_index, max_size):
    if not condition_image_paths:
        return None, 0, 0

    img_path = condition_image_paths[0]
    frame = Image.open(img_path).convert("RGB")
    frame = crop_and_resize(frame, max_size)
    width, height = frame.size

    frames = []
    for i in range(81):
        if i in condition_index:
            idx_in_list = condition_index.index(i)
            img_path = condition_image_paths[idx_in_list]
            frame = Image.open(img_path).convert("RGB")
            frame = crop_and_resize(frame, max_size)
        else:
            frame_tensor = torch.zeros((3, height, width), dtype=torch.uint8)
            frame = torchvision.transforms.functional.to_pil_image(frame_tensor)
        frames.append(frame)

    return frames, height, width


def preprocess_image(image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=torch_dtype, device=device or 'cuda')
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image

def calculate_needed_dilation(subject_image_path, skeleton_image_path):
    max_iterations = 100
    subject_img = cv2.imread(subject_image_path)
    skeleton_img = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)

    if subject_img is None or skeleton_img is None:
        print(f"Error loading images for dilation calculation: {subject_image_path}")
        return 1

    subject_img = resize_image(subject_img)
    skeleton_img = resize_image(skeleton_img)

    if subject_img.shape[:2] != skeleton_img.shape[:2]:
        subject_img = cv2.resize(subject_img, (skeleton_img.shape[1], skeleton_img.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

    subject_gray = cv2.cvtColor(subject_img, cv2.COLOR_BGR2GRAY)
    _, subject_mask = cv2.threshold(subject_gray, 245, 255, cv2.THRESH_BINARY_INV)
    subject_mask = np.where(subject_mask > 0, 255, 0).astype(np.uint8)
    _, skeleton_mask = cv2.threshold(skeleton_img, 127, 255, cv2.THRESH_BINARY)

    total_subject_pixels = cv2.countNonZero(subject_mask)
    if total_subject_pixels == 0:
        return 1

    current_dilated = skeleton_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    try:
        for i in range(0, max_iterations, max(1, max_iterations // 100)):
            intersection = cv2.bitwise_and(current_dilated, subject_mask)
            covered_pixels = cv2.countNonZero(intersection)
            coverage = covered_pixels / total_subject_pixels
            if coverage >= 1.0:
                return i + 1
            current_dilated = cv2.dilate(current_dilated, kernel, iterations=1)
    except Exception as e:
        print(f"Error in dilation calculation: {e}")

    return max_iterations

def segment_skeleton(skeleton_path, id_path=None, dilation_times=1):
    # 1. Binarize
    skeleton_image = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    if skeleton_image is None:
        raise ValueError(f"Cannot read skeleton image: {skeleton_path}")

    skeleton_image = resize_image(skeleton_image)
    _, binary = cv2.threshold(skeleton_image, 127, 255, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary // 255).astype(np.uint8)

    # 2. Subject mask
    subject_mask = None
    if id_path is not None and os.path.exists(id_path):
        id_image = cv2.imread(id_path)
        id_image = resize_image(id_image)
        subject_gray = cv2.cvtColor(id_image, cv2.COLOR_BGR2GRAY)
        _, subject_mask = cv2.threshold(subject_gray, 245, 255, cv2.THRESH_BINARY_INV)
        subject_mask = np.where(subject_mask > 0, 255, 0).astype(np.uint8)

    H, W = skeleton_image.shape
    h_latents, w_latents = skeleton.shape[:2]

    # 3. Find keypoints
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    endpoints, junctions = [], []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skeleton[y, x] == 1:
                count = sum(skeleton[y + dy, x + dx] == 1 for dy, dx in neighbors)
                if count == 1:
                    endpoints.append((x, y))
                elif count > 2:
                    junctions.append((x, y))

    key_points = set(endpoints + junctions)
    key_points_array = np.zeros_like(skeleton)
    for (x, y) in key_points: key_points_array[y, x] = 1
    skeleton_without_keys = skeleton - key_points_array
    skeleton_without_keys[skeleton_without_keys < 0] = 0

    # 4. BFS Split
    queue = deque()
    segment = np.zeros_like(skeleton)
    segment_id = 0
    segments = []
    for y in range(H):
        for x in range(W):
            if skeleton_without_keys[y, x] == 1 and segment[y, x] == 0:
                queue.append((x, y))
                segment[y, x] = segment_id + 1
                current_segment = [(x, y)]
                while queue:
                    cx, cy = queue.popleft()
                    for dy, dx in neighbors:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if skeleton_without_keys[ny, nx] == 1 and segment[ny, nx] == 0:
                                segment[ny, nx] = segment_id + 1
                                current_segment.append((nx, ny))
                                queue.append((nx, ny))
                if len(current_segment) > 1:
                    segments.append(current_segment)
                    segment_id += 1

    # 5. Merge segments
    def segments_distance(seg_a, seg_b):
        min_dist = float('inf')
        pts_a = np.array(seg_a)
        pts_b = np.array(seg_b)
        dists = np.linalg.norm(pts_a[:, None] - pts_b, axis=2)
        return np.min(dists)

    while len(segments) > 0 and (len(segments) > 5 or min(len(seg) for seg in segments) < 20):
        seg_lengths = [len(seg) for seg in segments]
        min_idx = np.argmin(seg_lengths)
        seg_a = segments[min_idx]

        min_dist = float('inf')
        merge_idx = -1

        for i, seg_b in enumerate(segments):
            if i == min_idx: continue
            dist = segments_distance(seg_a, seg_b)
            if dist < min_dist:
                min_dist = dist
                merge_idx = i

        if merge_idx == -1: break
        segments[merge_idx].extend(segments[min_idx])
        segments.pop(min_idx)

    # 6. Dilation
    segment_arrays = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for seg in segments:
        seg_array = np.zeros(skeleton_image.shape[:2], dtype=np.uint8)
        for (x, y) in seg: seg_array[y, x] = 255
        dilated_segment = cv2.dilate(seg_array, kernel, iterations=dilation_times)

        if subject_mask is not None:
            if dilated_segment.shape != subject_mask.shape:
                subject_mask = cv2.resize(subject_mask, (dilated_segment.shape[1], dilated_segment.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            dilated_segment = cv2.bitwise_and(dilated_segment, subject_mask)
        segment_arrays.append(dilated_segment)

    return segment_arrays, w_latents // 32, h_latents // 32


# -------------------------- Dataset 类 --------------------------

class VideoGenDataset(Dataset):
    def __init__(self, base_path, metadata_path, output_path):
        self.base_path = base_path
        self.metadata_path = metadata_path
        if not os.path.exists(self.metadata_path):
            self.metadata_path = os.path.join(self.base_path, metadata_path)

        self.data = self._load_and_filter_metadata(output_path)

    def _load_and_filter_metadata(self, output_dir):
        data = []
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file not found: {self.metadata_path}")
            return []

        with open(self.metadata_path, mode="r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                file_name = os.path.splitext(row["file_name"])[0]
                output_path = os.path.join(output_dir, f"{file_name}.mp4")
                condition_index = ast.literal_eval(row["condition_index"])
                data.append({
                    "video_path": row["file_name"],
                    "file_name": file_name,
                    "condition_images": [f"{i:03d}.png" for i in condition_index],
                    "condition_index": condition_index,
                    "input_image_path": os.path.join(self.base_path, "first_frame", f"{file_name}.png"),
                    "id_path": os.path.join(self.base_path, "video", f"{file_name}_id.png"),
                    "output_path": output_path,
                    "text": row["text"]
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import cv2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from PIL import Image
import data_utils

def process_single_sample(base_path, data, pipe, device, rank, max_size):
    try:
        file_name = data["file_name"][0]
        video_path = data["video_path"][0]
        condition_images_files = [img[0] for img in data["condition_images"]]
        condition_index = [t.item() for t in data["condition_index"]]
        input_image_path = data["input_image_path"][0]
        output_path = data["output_path"][0]
        prompt = data["text"][0]

        input_image = Image.open(input_image_path)
        if input_image is None:
            raise ValueError(f"[rank{rank}] input image does not exits：{input_image_path}")
        input_image = data_utils.crop_and_resize(input_image, max_size)

        valid_indices = [i for i, idx in enumerate(condition_index) if idx < 81]
        condition_images_files = [condition_images_files[i] for i in valid_indices]
        condition_index = [condition_index[i] for i in valid_indices]

        full_skeleton_paths = [os.path.join(base_path, "skeleton_image", file_name, f) for f in condition_images_files]

        condition_images, height, width = data_utils.load_condition_image(full_skeleton_paths, condition_index,
                                                                          max_size)
        if condition_images is None or height == 0:
            raise ValueError(f"[rank{rank}] load skeleton fail：{video_path}")

        id_path = os.path.join(base_path, "video", f"{file_name}_id.png")
        alpha = data_utils.calculate_needed_dilation(id_path, full_skeleton_paths[0])

        binary_masks = []
        w_latents, h_latents = 0, 0

        for i, path in enumerate(full_skeleton_paths):
            current_id_path = id_path if i == 0 else None

            seg_masks, w, h = data_utils.segment_skeleton(path, current_id_path, alpha)

            if i == 0:
                w_latents, h_latents = w, h

            processed_seg_masks = []
            for mask in seg_masks:
                resized_mask = cv2.resize(mask, (w_latents, h_latents), interpolation=cv2.INTER_NEAREST)
                normalized_mask = (resized_mask > 127).astype(np.float32)
                processed_seg_masks.append(normalized_mask)

            binary_masks.append(processed_seg_masks)

        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=input_image,
            condition_images=condition_images,
            condition_index=condition_index,
            binary_masks=binary_masks,
            pose_cfg=False,
            num_inference_steps=50,
            cfg_scale=1.0,
            seed=123,
            height=height,
            width=width,
            tiled=True,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )

        save_video(video, output_path, fps=24, quality=5)
        print(f"[rank {rank}] generates：{output_path} complete \n")

    except Exception as e:
        error_msg = f"[rank {rank}] fails in generating：{file_name if 'file_name' in locals() else 'Unknown'} | error：{str(e)}\n"


def get_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0)
        args, _ = parser.parse_known_args()
        return args.local_rank

def main():
    parser = argparse.ArgumentParser(description='Pose Anything Generation Script')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to the metadata CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated outputs')
    parser.add_argument('--base_path', type=str, required=True, help='Root directory path of the input dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the generation model weights')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the pretrained model directory')
    parser.add_argument('--max_size', type=int, default=832, help='Maximum length of the longest side for input images')
    args = parser.parse_args()

    local_rank = get_local_rank()

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    np.random.seed(42 + rank)

    if rank == 0:
        print(f"Total Processes: {world_size} | Model Path: {args.model_path} | Metadata Path: {args.metadata_path}")
        print(f"Base Path: {args.base_path}")

    os.makedirs(args.output_path, exist_ok=True)
    dataset = data_utils.VideoGenDataset(args.base_path, args.metadata_path, args.output_path)

    if rank == 0:
        filtered_len = len(dataset)
    else:
        filtered_len = 0

    broadcast_list = [filtered_len]
    dist.broadcast_object_list(broadcast_list, src=0)
    filtered_len = broadcast_list[0]

    if rank != 0:
        dataset.data = dataset.data[:filtered_len]

    sampler = DistributedSampler(
        dataset,
        shuffle=False,
        rank=rank,
        num_replicas=world_size
    )
    sampler.set_epoch(0)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    t5_path = os.path.join(args.pretrained_path, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(args.pretrained_path, "Wan2.2_VAE.pth")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                path=t5_path,
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu"),
            ModelConfig(path=args.model_path),
            ModelConfig(
                path=vae_path,
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu"),
        ],
    )

    pipe.enable_vram_management()
    print(f"[rank {rank}] model loading success！")

    for batch in dataloader:
        process_single_sample(args.base_path, batch, pipe, device, rank, args.max_size)

    dist.barrier()
    if rank == 0:
        print("=== All Generation Complete ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
import os
import torch
from safetensors.torch import load_file, save_file

file_paths = [
    "./models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "./models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "./models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
]

def resize_and_save_patch_embedding(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"file does not exist: {input_path}")
        return

    try:
        tensors = load_file(input_path)
        if "patch_embedding.weight" not in tensors:
            print(" 'patch_embedding.weight' does not exist")
            return

        original_weight = tensors["patch_embedding.weight"]
        if original_weight.shape != (3072, 48, 1, 2, 2):
            print(f"Error：original shape {original_weight.shape}，expected to be (3072, 48, 1, 2, 2)")
            return

        new_shape = (3072, 96, 1, 2, 2)
        new_weight = torch.zeros(new_shape, dtype=original_weight.dtype, device=original_weight.device)
        new_weight[:, :48, :, :, :] = original_weight
        tensors["patch_embedding.weight"] = new_weight

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        backup_path = output_path.replace(".safetensors", "-backup.safetensors")
        if os.path.exists(output_path):
            print(f"backup origin checkpoint to: {backup_path}")
            os.rename(output_path, backup_path)

        save_file(tensors, output_path)
        print(f"save to: {output_path}")
        print(f"new shape: {new_weight.shape}")

    except Exception as e:
        print(f"Error: {e}")

input_file = file_paths[0]
output_file = file_paths[0]

resize_and_save_patch_embedding(input_file, output_file)
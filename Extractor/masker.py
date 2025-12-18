#Grounded-SAM-2
import argparse
import supervision as sv
from Grounded.sam2.build_sam import build_sam2
from Grounded.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from Grounded.utils.supervision_utils import CUSTOM_COLOR_MAP
import time
import os
import numpy as np
import torch
import cv2
from PIL import Image

"write error into log"
def write_error_log(log_path, message):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")


FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./Grounded/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
error_log_path = "./log.txt"

def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def calculate_iou(bbox1, bbox2):

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_inter = max(x1_1, x1_2)
    x2_inter = min(x2_1, x2_2)
    y1_inter = max(y1_1, y1_2)
    y2_inter = min(y2_1, y2_2)
    
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        intersection_area = 0
    else:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0  
    else:
        return intersection_area / union_area
    
def find_best_matching_bbox(result, ref_box, ref_label):

    od_results = result.get('<OD>', {})
    bboxes = od_results.get('bboxes', [])
    labels = od_results.get('labels', [])
    
    max_iou = 0.0
    best_bbox = None

    for bbox, label in zip(bboxes, labels):
        iou=0.0
        if label == ref_label and ref_box is not None:
            iou = calculate_iou(bbox, ref_box)
        elif label==ref_label and ref_box is None:
            iou = calculate_area(bbox)
        if iou > max_iou or max_iou == 0.0:
            max_iou = iou
            best_bbox = bbox
    
    return best_bbox, ref_label


def find_best_bbox(result, ref_box):
    od_results = result.get('<OD>', {})
    bboxes = od_results.get('bboxes', [])
    labels = od_results.get('labels', [])

    max_iou = 0.0
    best_bbox = None
    best_label = None

    for bbox, label in zip(bboxes, labels):
        iou = 0.0
        if ref_box is not None:
            iou = calculate_iou(bbox, ref_box)
        elif ref_box is None:
            iou = calculate_area(bbox)
        if iou > max_iou:
            max_iou = iou
            best_bbox = bbox
            best_label = label

    return best_bbox, best_label


def get_largest_bbox(results):
    
    if '<OD>' not in results or 'bboxes' not in results['<OD>'] or 'labels' not in results['<OD>']:
        return None, None
    
    bboxes = results['<OD>']['bboxes']
    labels = results['<OD>']['labels']
    
    if not bboxes or not labels:
        return None, None
    
    max_area = 0
    max_index = 0
    print(labels)
    for i, bbox in enumerate(bboxes):
        area = calculate_area(bbox)
        if area >= max_area or max_area == 0:
            max_area = area
            max_index = i
    return bboxes[max_index], labels[max_index]


def run_florence2(task_prompt, text_input, model, processor, image):

    assert model is not None, "not initialize Florence-2 model"
    assert processor is not None, "not initialize Florence-2 processor"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer


class MaskGenerator:
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("loading Florence-2 ...")
        self.florence2_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL_ID, 
            trust_remote_code=True, 
            torch_dtype='auto',
        ).eval().to(self.device)
        self.florence2_processor = AutoProcessor.from_pretrained(
            FLORENCE2_MODEL_ID, 
            trust_remote_code=True
        )
        
        # Initialize SAM 2
        print("loading SAM 2 ...")
        self.sam2_predictor = SAM2ImagePredictor(
            build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self.device)
        )
        print("loading success")


    def object_detection_and_segmentation(self, image_input, output_path, ref_box=None, ref_label=None):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 加载图像
            try:
                if isinstance(image_input, Image.Image):
                    image=image_input
                else:
                    image = Image.open(str(image_input)).convert("RGB")
            except Exception as e:
                error_msg = f"read {image_input} error: {str(e)}"
                print(error_msg)
                return None, None
            image_np = np.array(image)
            origin_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            task_prompt = "<OD>"
            text_input = None
            results = run_florence2(
                task_prompt, 
                text_input, 
                self.florence2_model, 
                self.florence2_processor, 
                image
            )

            if ref_box is not None and ref_label is None:
                best_bbox, best_label= find_best_bbox(results, ref_box)
            elif ref_box is None and ref_label is None:
                best_bbox, best_label = get_largest_bbox(results)
            else:
                best_bbox, best_label = find_best_matching_bbox(results, ref_box, ref_label)
                
            if best_bbox is not None and best_label is not None:
                results['<OD>']['bboxes'] = [best_bbox]
                results['<OD>']['labels'] = [best_label]
            else:
                results['<OD>'] = {'bboxes': [], 'labels': []}

                return best_bbox, ref_label

            results = results[task_prompt]
            input_boxes = np.array(results["bboxes"])
            class_names = results["labels"]

            self.sam2_predictor.set_image(np.array(image))
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            if masks.ndim == 4:
                masks = masks.squeeze(1)
            mask = masks[0].astype(bool) if masks.ndim == 3 else masks.astype(bool)
            mask_3d = np.stack([mask] * 3, axis=-1)
            
            masked_img = np.where(mask_3d, origin_image, 255)

            try:
                cv2.imwrite(output_path, masked_img)
            except Exception as e:
                error_msg = f"save {output_path} fail: {str(e)}"
                write_error_log(error_log_path, error_msg)
                
            return best_bbox, best_label
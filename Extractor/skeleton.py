import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

np.bool8 = np.bool_
np.float_ = np.float64
np.obj2sctype = lambda obj: np.dtype(obj).type


from BlumNet.datasets import build_dataset
from BlumNet.models import build_model
from BlumNet.reconstruction import PostProcess
import BlumNet.lib.misc as utils


def write_error_log(log_path, message):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")


class Skeletonizer:

    def __init__(self, args):
        self.args = args
        self.resume= args.resume
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._set_seed()
        self.model = self._build_model()
        self.postprocessor = PostProcess(eval_score=0.5)
        self._load_checkpoint()


    def _set_seed(self):
        seed = 42 + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


    def _build_model(self):
        model = build_model(self.args)[0]
        model.to(self.device)
        model.eval()
        return model

    def _load_checkpoint(self):
        if not self.resume:
            raise ValueError(f"Model weights path not specified.: resume={self.resume}")
        try:
            checkpoint = torch.load(self.resume, map_location='cpu',weights_only=False)
            self.model.load_state_dict(checkpoint['model'], strict=True)
        except Exception as e:
            raise RuntimeError(f"loading fail: {str(e)}")

    def skeletonization(self, input_dir, output_dir):
        try:
            img_dir = Path(input_dir).name
            self.args.data_root = input_dir

            try:
                dataset_val = build_dataset(image_set='infer', args=self.args)
            except Exception as e:
                error_msg = f"construct {input_dir} fail: {str(e)}"
                print(error_msg)
                return

            len_imgs = len(dataset_val)
            for ii in range(len_imgs):
                try:
                    img, target = dataset_val[ii]
                    inputName, _ = dataset_val.id2name(ii)
                    raw_img = Image.open(inputName).convert("RGB")
                    file_name = os.path.splitext(os.path.basename(inputName))[0]

                    # (PIL RGB -> OpenCV BGR)
                    _raw_img = np.array(raw_img)[:, :, ::-1]  # RGB to BGR
                    vis_img = np.copy(_raw_img)
                    imgs = img[None, ...].to(self.device)
                    targets = [{k: v.to(self.device) for k, v in target.items()}]

                    with torch.no_grad():
                        outputs = self.model(imgs)

                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    results_dict = self.postprocessor(
                        outputs,
                        orig_target_sizes,
                        ignore_graph=("lines" in self.args.visual_type)
                    )

                    pred = results_dict['curves'][0]
                    ptspred = results_dict['pts'][0]
                    graphs = results_dict.get('graphs', None)

                    black_canvas = np.zeros_like(_raw_img)

                    # draw line and dot
                    if "lines" in self.args.visual_type:
                        black_canvas, _ = self.postprocessor.visualise_curves(
                            pred, 0.65, black_canvas, thinning=True
                        )
                    elif graphs is not None:
                        for _, (branches, _) in graphs[0].items():
                            branches = [np.int32(b) for b in branches]
                            for b in branches:
                                cv2.polylines(
                                    black_canvas,
                                    [b],
                                    False,
                                    color=(255, 255, 255),  # 白色线条
                                    thickness=3
                                )

                    black_canvas, _ = self.postprocessor.visualise_pts(
                        ptspred, 0.05, black_canvas
                    )

                    output_subdir = os.path.join(output_dir, img_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, f"{file_name}.png")
                    cv2.imwrite(output_path, black_canvas)

                except Exception as e:
                    error_msg = f"process {inputName} fail: {str(e)}"
                    tqdm.write(error_msg)
                    continue

        except Exception as e:
            error_msg = f"skeletonizatio fail: {str(e)}"
            print(error_msg)


def run_skeletonization(args, intput_dir, output_dir):

    if not hasattr(args, 'visual_type'):
        args.visual_type = "branches"

    skeletonizer = Skeletonizer(args)
    skeletonizer.skeletonization(intput_dir,output_dir)
    print(f"skeleton done, save to: {output_dir}")
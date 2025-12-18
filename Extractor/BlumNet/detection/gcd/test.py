import argparse
import random
import time
import cv2
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from args_parser import get_args_parser
from datasets import build_dataset
from models import build_model
from preparation import CONNECT_PT, JUNCTION_PT, END_PT
from reconstruction import PostProcess
import lib.misc as utils


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)
    postprocessor = PostProcess(eval_score=args.eval_score)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_val = build_dataset(image_set='val', args=args) # train val infer

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError(f'resume error: {args.resume}')
    args.eval = True

    model.eval()
    criterion.eval()

    dataname = args.dataset_file
    visual_dir = Path(f'{args.data_root}/{dataname}/test/{args.resume.split("/")[-2]}_vis')
    visual_dir.mkdir(exist_ok=True, parents=True)
    visual_type = args.visual_type # "lines" "branches"

    len_imgs = len(dataset_val)
    start_time = time.time()
    postpro_time = 0
    for ii in range(len_imgs):
        img, target = dataset_val[ii]
        inputName, targetName = dataset_val.id2name(ii)
        raw_img = Image.open(inputName).convert("RGB")

        # raw_skeleton = (cv2.imread(targetName, 1) > 0).astype(np.uint8) * 255
        _raw_img = np.array(raw_img)[:, :, ::-1]

        imgs = img[None, ...].to(device)
        targets = [{k: v.to(device) for k, v in target.items()}]

        outputs = model(imgs)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        t1 = time.time()
        results_dict = postprocessor(outputs, orig_target_sizes, ignore_graph=("lines" in visual_type))
        t2 = time.time()
        postpro_time += (t2 - t1)

        pred = results_dict['curves'][0]
        ptspred = results_dict['pts'][0]
        graphs = results_dict.get('graphs', None)

        vis_img = np.copy(_raw_img)
        # ========= draw lines
        if "lines" in visual_type:
            vis_img, curves_mask = postprocessor.visualise_curves(pred, 0.65, vis_img, thinning=True, ch3mask=True, vmask=255)
        # ========== draw graphs
        elif graphs is not None:
            for _, (branches, _) in graphs[0].items():
                branches = [np.int32(b) for b in branches]
                for b in branches:
                    color = (random.randint(150, 255), random.randint(100, 255), random.randint(100, 255))
                    cv2.polylines(vis_img, [b], False, color=color, thickness=3) 
        vis_img, pts_mask = postprocessor.visualise_pts(ptspred, 0.05, vis_img)

        base_name = os.path.basename(inputName)
        cv2.imwrite(
            f"{str(visual_dir)}/{base_name.split('.')[0]}_{visual_type}.png",
            np.concatenate([vis_img, curves_mask], axis=1)
        )

    diff_time = time.time() - start_time
    print('Detection took {:.3f}s per image, including postprocess {:.3f}s per image'.format(
        diff_time / len_imgs, postpro_time / len_imgs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('visualise script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

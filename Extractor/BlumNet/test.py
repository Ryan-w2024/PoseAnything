import argparse
import glob
import random
import time
import cv2
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datasets import build_dataset
from models import build_model
from reconstruction import PostProcess
from detection.gcd.args_parser import get_args_parser
import lib.misc as utils


def main(args, save_dir):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)[0]
    model.to(device)
    postprocessor = PostProcess(eval_score=args.eval_score)
    dataset_val = build_dataset(image_set='infer', args=args)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        raise ValueError(f'resume error: {args.resume}')
    args.eval = True
    model.eval()

    visual_type = args.visual_type
    len_imgs = len(dataset_val)
    start_time = time.time()
    for ii in range(len_imgs):
        img, target = dataset_val[ii]
        inputName, _ = dataset_val.id2name(ii)
        raw_img = Image.open(inputName).convert("RGB")
        _raw_img = np.array(raw_img)[:, :, ::-1]
        vis_img = np.copy(_raw_img)
        imgs = img[None, ...].to(device)
        targets = [{k: v.to(device) for k, v in target.items()}]

        outputs = model(imgs)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_dict = postprocessor(outputs, orig_target_sizes, ignore_graph=("lines" in visual_type))

        pred = results_dict['curves'][0]
        ptspred = results_dict['pts'][0]
        graphs = results_dict.get('graphs', None)

        # ========= draw lines&pts
        if "lines" in visual_type:
            vis_img, _ = postprocessor.visualise_curves(pred, 0.65, vis_img, thinning=True)
        # ========== draw branches
        elif graphs is not None:
            for _, (branches, _) in graphs[0].items():
                branches = [np.int32(b) for b in branches]
                for b in branches:
                    color = (random.randint(150, 255), random.randint(100, 255), random.randint(100, 255))
                    cv2.polylines(vis_img, [b], False, color=color, thickness=3) 
        vis_img, _ = postprocessor.visualise_pts(ptspred, 0.05, vis_img)
        base_name = os.path.basename(inputName).split('.')[0]
        cv2.imwrite(f"{save_dir}/{base_name}_{visual_type}.jpg", vis_img)

    diff_time = time.time() - start_time
    print('Detection took {:.3f}s per image'.format(diff_time / len_imgs))


def run_demo_img(args):
    args.data_root = "./datasets/demo_imgs"
    args.visual_type = "branches" # "lines&pts" # 
    main(args, save_dir='./datasets/demo_rst')


def run_demo_video(args):
    args.visual_type = "branches" # "lines&pts" # 
    v_name = 'horse.mp4'
    tmp_root = Path("./datasets/tmp/")

    # step 1: video to images
    cap = cv2.VideoCapture()
    vfileid = v_name.split('.')[0]
    vroot = tmp_root / vfileid
    vroot.mkdir(exist_ok=True, parents=True)
    cap.open(f"./datasets/demo_video/{v_name}")
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(vroot / f"{cnt}.png"), frame)
        cnt += 1
    cap.release()

    # step 2: find skeletons for images
    save_tmp_dir = Path(f'./datasets/demo_rst/{vfileid}')
    save_tmp_dir.mkdir(exist_ok=True, parents=True)
    args.data_root = f"./datasets/tmp/{vfileid}"
    main(args, save_dir=str(save_tmp_dir))

    # step3: reconstruct gif demo
    gifs = []
    fnames = glob.glob(str(save_tmp_dir / "*branch*jpg"))
    fnames.sort()
    for fname in fnames:
        frame = cv2.imread(fname, 1)
        gifs.append(Image.fromarray(frame[:,:,::-1]))
    gifs[0].save(
        f"{str(save_tmp_dir)}.gif",
        format='GIF',
        append_images=gifs[1::],
        save_all=True,
        duration=1,
        loop=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('visualise script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.resume = "exps/demo/checkpoint.pth"
    args.num_feature_levels = 3
    args.aux_loss = True
    args.gid = True
    args.out_pts=128
    # run_demo_img(args)
    run_demo_video(args)

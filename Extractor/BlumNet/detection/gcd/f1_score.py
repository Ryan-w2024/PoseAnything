import argparse
import cv2
import random
import time
import torch
import numpy as np
import lib.misc as utils
from pathlib import Path
from args_parser import get_args_parser
from datasets import build_dataset
from models import build_model
from reconstruction import PostProcess
from lib.libmetric import SkeletonEvaluator


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
    postprocessor = PostProcess(eval_score=None)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_val = build_dataset(image_set='test', args=args)

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
    len_imgs = len(dataset_val)
    diff_time, infer_time, postpro_time = 0, 0, 0
    eval_tools = [[SkeletonEvaluator(), sc] for sc in
        np.arange(0.02, 0.09, 0.02).tolist() + \
        np.round(np.arange(0.1, 0.96, 0.05), 2).tolist() + \
        np.arange(0.96, 0.99, 0.01).tolist()
    ]
    # eval_tools = [[SkeletonEvaluator(), 0.65]]
    for ii in range(len_imgs):
        t1 = time.time()
        img, target = dataset_val[ii]
        w, h = target['orig_size'].data.cpu().numpy()

        im_path, gt_path = dataset_val.id2name(ii)
        gt_skeleton = (cv2.imread(gt_path, 0) > 0).astype(np.uint8) * 255

        imgs = img[None, ...].to(device)
        targets = [{k: v.to(device) for k, v in target.items()}]

        outputs = model(imgs)
        t2 = time.time()

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_dict = postprocessor(outputs, orig_target_sizes, ignore_graph=True)
        pred = results_dict['curves'][0]
        for sk_evaluator, eval_score in eval_tools:
            _, pred_mask = postprocessor.visualise_curves(
                pred, eval_score, np.zeros((h, w, 3), dtype=np.uint8))
            sk_evaluator.update([(gt_skeleton, pred_mask, im_path)])
        t3 = time.time()
        infer_time += (t2 - t1)
        postpro_time += ((t3 - t2) / len(eval_tools))
        diff_time += ((t3 - t2) / len(eval_tools) + (t2 - t1))
    print('Detection took {:.3f}s per image: Inference {:.3f}s + Postprocess {:.3f}s'.format(
        diff_time / len_imgs, infer_time / len_imgs, postpro_time / len_imgs))

    # accumulate predictions from all images
    if len(eval_tools) == 1:
        visual_dir = Path(f'{args.data_root}/{dataname}/test/{args.resume.split("/")[-2]}')
        visual_dir.mkdir(exist_ok=True, parents=True)
        visual_dir = str(visual_dir)
    else:
        visual_dir = None
    print("Metrics@offset_threshold=0.01")
    print("score_thrsh\tcnt_recall\tcnt_precision\tf1-score")
    for sk_evaluator, eval_score in eval_tools:
        if len(eval_tools) == 1:
            sk_evaluator.vis_only(visual_dir)
            break
        metrics = sk_evaluator.summarize_cum(score_threshold=eval_score, offset_threshold=0.01, visual_dir=visual_dir)
        print(f"{eval_score}\t{metrics['cnt_recall']}\t{metrics['cnt_precision']}\t{metrics['cnt_f1']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

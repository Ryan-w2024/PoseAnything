# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path


def read_im_gt(imp, gtp):
    im = cv2.imread(imp, 1)
    gt = 255 * (cv2.imread(gtp, 0) > 0).astype(np.uint8)
    h, w = im.shape[:2]
    if h < w:
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
    return im, gt


def stitch_objs(imp1, imp2, gtp1, gtp2):
    im1, gt1 = read_im_gt(imp1, gtp1)
    im2, gt2 = read_im_gt(imp2, gtp2)
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    if h1 < h2:
        (im1, gt1, im2, gt2) = (im2, gt2, im1, gt1)
        (h1, w1, h2, w2) = (h2, w2, h1, w1)
    nw2 = int(h1 * 1.0 / h2 * w2)
    im2 = cv2.resize(im2, dsize=(nw2, h1))
    gt2 = cv2.resize(gt2, dsize=(nw2, h1))
    im_stitch = np.concatenate([im1, im2], axis=1)
    gt_stitch = np.concatenate([gt1, gt2], axis=1)
    mask1, mask2 = 100 * np.ones_like(gt1), 250 * np.ones_like(gt2)
    mask_stitch = np.concatenate([mask1, mask2], axis=1)
    return im_stitch, gt_stitch, mask_stitch


def id2name(id, root, frame):
    image_fpath = os.path.join(root, frame.iloc[id, 0])
    ann_fpath = os.path.join(root, frame.iloc[id, 1])
    return image_fpath, ann_fpath


def create_stitches(root, lst_txt):
    lst_path = f"{root}/{lst_txt}"
    frame = pd.read_csv(lst_path, dtype=str, delimiter=' ', header=None)
    im_nums = len(frame)
    lst_info = lst_txt.split('/')
    lst_info[0] = f"{lst_info[0]}_mix"
    new_lst_path = "/".join(lst_info)
    stitch_dir = Path(f"{root}/{lst_info[0]}/{lst_info[1]}")
    (stitch_dir / 'im').mkdir(exist_ok=True, parents=True)
    (stitch_dir / 'gt').mkdir(exist_ok=True, parents=True)
    with open(f"{root}/{new_lst_path}", "w") as ff:
        for i in range(im_nums):
            j = random.randint(0, im_nums - 1)
            imp1, gtp1 = id2name(i, root, frame)
            imp2, gtp2 = id2name(j, root, frame)
            im_stitch, gt_stitch, mask_stitch = stitch_objs(imp1, imp2, gtp1, gtp2)
            fileid1 = imp1.split("/")[-1][:-4]
            fileid2 = imp2.split("/")[-1][:-4]
            rst_imp = f"{fileid1}_{fileid2}.jpg"
            rst_gtp = f"{fileid1}_{fileid2}.png"
            rst_maskp = f"{fileid1}_{fileid2}_mask.png"
            ff.write(f"{lst_info[0]}/{lst_info[1]}/im/{rst_imp} {lst_info[0]}/{lst_info[1]}/gt/{rst_gtp}\n")
            cv2.imwrite(f"{root}/{lst_info[0]}/{lst_info[1]}/im/{rst_imp}", im_stitch)
            cv2.imwrite(f"{root}/{lst_info[0]}/{lst_info[1]}/gt/{rst_gtp}", gt_stitch)
            cv2.imwrite(f"{root}/{lst_info[0]}/{lst_info[1]}/gt/{rst_maskp}", mask_stitch)




if __name__ == '__main__':

    for ttype in ['test', 'train']:
        root = "./datasets"
        lst_txt = f"sk1491/{ttype}/{ttype}_pair.lst"
        create_stitches(root, lst_txt)


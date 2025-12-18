# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import numpy as np

def get_rotation_homo(h, w, angle):
    (cX, cY) = (0, 0)
    pts = np.float32(
        [[0, 0, 1],
         [w, 0, 1],
         [w, h, 1],
         [0, h, 1]]
    )
    M = np.zeros((3, 3), dtype=np.float32)
    M[:2, :] = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    M[2, 2] = 1.
    new_pts = np.matmul(np.mat(M), np.mat(pts).H).A
    new_pts /= new_pts[2]
    xmin, ymin = np.min(new_pts[:2, :], axis=1).astype(np.int32).tolist()
    xmax, ymax = np.max(new_pts[:2, :], axis=1).astype(np.int32).tolist()
    M[0, 2] = -xmin
    M[1, 2] = -ymin
    nw, nh = xmax - xmin, ymax - ymin
    affine_homo = M[:2,:]
    return affine_homo, (nw, nh)

def rotate_image(img, ang, back=255):
    h, w = img.shape[:2]
    M, (nw, nh) = get_rotation_homo(h, w, ang)
    if len(img.shape) == 3:
        dst = np.ones(shape=(nh, nw, 3), dtype=np.uint8) * back
    elif len(img.shape) == 2:
        dst = np.ones(shape=(nh, nw), dtype=np.uint8) * back
    else:
        raise NotImplementedError('')
    cv2.warpAffine(img, M, (nw, nh), dst, borderMode=cv2.BORDER_CONSTANT, borderValue=(back, back, back))
    return dst

def vhflip(img, flipCode=0):
    # flipCode:
    #   >0: (right-left),
    #   0: (bottom-up),
    #   <0: both right-left and bottom-up
    return cv2.flip(img, flipCode=flipCode)


def multi_test_ssleaves(src_dir, dst_root, back=255, src_gt=None, dst_gt=None):
    from pathlib import Path
    dst_types = ['4rots', '8rots', 'vhflip', 'scales']
    for dst_type in dst_types:
        dst_dir = f'{dst_root}/test_{dst_type}'
        Path(dst_dir).mkdir(exist_ok=True, parents=True)
        if src_gt:
            (Path(dst_gt) / f'test_{dst_type}').mkdir(exist_ok=True, parents=True)

    for i in Path(src_dir).iterdir():
        filename = i.name.split('.')[0]
        raw = cv2.imread(str(i), 1)
        h, w = raw.shape[:2]
        raw_gt = cv2.imread(f"{src_gt}/{filename}.png", 1) if src_gt else None
        # remove the black box at raw image's boundary
        nimg = 254 * np.ones(shape=(h, w, 3), dtype=np.uint8)
        nimg[15:h - 15, 15:w - 15] = raw[15:h - 15, 15:w - 15]

        # =========== For rotation of 0-90-180-270 angles
        for angle in [0, 90, 180, 270]:
            rst = rotate_image(nimg, angle, back=back)
            cv2.imwrite(
                f"{dst_root}/test_{dst_types[0]}/{filename}_rotate_{angle}.png",
                rst)
            if raw_gt is not None:
                rst_gt = rotate_image(raw_gt, angle, back=0)
                cv2.imwrite(
                    f"{dst_gt}/test_{dst_types[0]}/{filename}_rotate_{angle}.png",
                    rst_gt)

        # For rotation of any angles
        for ii in range(8):
            any_angle = np.random.randint(-179, 179)
            rst = rotate_image(nimg, any_angle, back=back)
            cv2.imwrite(
                f"{dst_root}/test_{dst_types[1]}/{i.name.split('.')[0]}_rotate_{ii}.png",
                rst)
            if raw_gt is not None:
                rst_gt = rotate_image(raw_gt, any_angle, back=0)
                cv2.imwrite(
                    f"{dst_gt}/test_{dst_types[1]}/{i.name.split('.')[0]}_rotate_{ii}.png",
                    rst_gt)

        # =========== For flip:
        # 0 raw,
        # 1 both right-left and bottom-up,
        # 2 bottom-up,
        # 3 right-left
        for ii in range(4):
            rst = vhflip(nimg, flipCode=ii-2) if ii != 0 else nimg
            cv2.imwrite(
                f"{dst_root}/test_{dst_types[2]}/{i.name.split('.')[0]}_flip_{ii}.png",
                rst)
            if raw_gt is not None:
                rst_gt = vhflip(raw_gt, flipCode=ii-2) if ii != 0 else raw_gt
                cv2.imwrite(
                    f"{dst_gt}/test_{dst_types[2]}/{i.name.split('.')[0]}_flip_{ii}.png",
                    rst_gt)
        # =========== For scales
        for scl in [2, 1.2, 0.7, 1]:
            rst = cv2.resize(nimg, dsize=None, fx=scl, fy=scl)
            cv2.imwrite(
                f"{dst_root}/test_{dst_types[3]}/{i.name.split('.')[0]}_scale_{int(scl*100)}.png",
                rst)
            if raw_gt is not None:
                rst_gt = cv2.resize(raw_gt, dsize=None, fx=scl, fy=scl)
                cv2.imwrite(
                    f"{dst_gt}/test_{dst_types[3]}/{i.name.split('.')[0]}_scale_{int(scl*100)}.png",
                    rst_gt)
    return

import os
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def combination_trans(img_p, gt_p):
    im = cv2.imread(img_p, 1)
    gt = cv2.imread(gt_p, 0)



if __name__=='__main__':


    src_dir = os.path.join(cur_dir, 'SmithsonianLeaves/train/im')
    dst_root = os.path.join(cur_dir, 'SmithsonianLeaves/train/im_aug')
    # multi_test_ssleaves(src_dir, dst_root)

    gt_dir = os.path.join(cur_dir, 'SmithsonianLeaves/train/gt')
    dst_gt = os.path.join(cur_dir, 'SmithsonianLeaves/train/gt_aug')
    multi_test_ssleaves(src_dir, dst_root, src_gt=gt_dir, dst_gt=dst_gt)


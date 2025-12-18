# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import random
import numbers
import torch
import torchvision.transforms.functional as F
import numpy as np
from BlumNet.lib.libcommon.rotate_any_angle import get_rotation_homo
from PIL import Image


branch_info = [
    "curves", # "branch",
]
sk_info = [
    'key_pts', #'skpts', 'points'
]

def vhflip(sample, flip_fun, scl, bias):
    sample['image'] = flip_fun(sample['image'])
    sample['skeleton'] = flip_fun(sample['skeleton'])
    for sk_key in sk_info:
        if sk_key in sample:
            sample[sk_key] = sample[sk_key] * scl + bias

    for per_branch in sample['branches']:
        for k in branch_info:
            if k not in per_branch:
                continue
            _pts = per_branch[k]
            per_branch[k] = _pts * scl + bias

    return sample


def hflip(sample):
    flip_fun = F.hflip
    scl = torch.as_tensor([-1, 1])
    bias = torch.as_tensor([1, 0])
    return vhflip(sample, flip_fun, scl, bias)


def vflip(sample):
    flip_fun = F.vflip
    scl = torch.as_tensor([1, -1])
    bias = torch.as_tensor([0, 1])
    return vhflip(sample, flip_fun, scl, bias)


def rotate_anly(sample, angle):
    assert angle >= -180 and angle <= 180
    if abs(angle) <= 0.1:
        return sample
    sample['image'] = sample['image'].rotate(
        -angle, resample=Image.BILINEAR, expand=True, fillcolor=(255, 255, 255)) # negative is clockwise
    sample['skeleton'] = sample['skeleton'].rotate(
        -angle, resample=Image.BILINEAR, expand=True, fillcolor=0)
    affine_homo, (nw, nh) = get_rotation_homo(1, 1, angle=angle) # positive is clockwise
    sample['orig_size'] = sample['image'].size

    for sk_key in sk_info:
        if sk_key in sample:
            skpts = sample[sk_key].numpy()
            tmp_pts = np.ones((skpts.shape[0], 3), dtype=np.float32)
            tmp_pts[:, :2] = tmp_pts[:, :2] * skpts
            sample[sk_key] = torch.as_tensor(np.matmul(affine_homo, np.mat(tmp_pts).T).T.A)

    for per_branch in sample['branches']:
        for k in branch_info:
            if k not in per_branch:
                continue
            _pts = per_branch[k].numpy()
            l, n, c = _pts.shape

            pts = np.ones((l * n, 3), dtype=np.float32)
            pts[:, :2] = _pts.reshape((l * n, 2)) * pts[:, :2]
            pts = torch.as_tensor(np.matmul(affine_homo, np.mat(pts).T).T.A)
            per_branch[k] = pts.reshape((l, n, c))

    return sample


def rotate90(sample, N):
    assert N in [0, 1, 2, 3]
    if N in [0, 1, 2]:
        return rotate_anly(sample, 90*N)
    else:
        return rotate_anly(sample, -90)


def resize(sample, size):
    w, h = sample['image'].size
    scl = size * 1.0 / max(w, h)
    new_size = (round(scl * h), round(scl * w))
    sample['image'] = F.resize(sample['image'], new_size)
    sample['skeleton'] = F.resize(sample['skeleton'], new_size)
    return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return hflip(sample)
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return vflip(sample)
        return sample


class RandomRotate90(object):
    def __init__(self,):
        pass

    def __call__(self, sample):
        return rotate90(sample, np.random.randint(4))


class RandomRotateAny(object):
    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return rotate_anly(sample, np.random.randint(-45, 45))
        return sample


class RandomResize(object):
    def __init__(self, sizes):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, sample):
        size = random.choice(self.sizes)
        return resize(sample, size)


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):
        img = sample['image']
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)
        sample['image'] = img
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample['id'] = torch.as_tensor(sample['id'])
        sample['orig_size'] = torch.as_tensor(sample['orig_size'])
        sample['image'] = F.to_tensor(sample['image'])
        sample['skeleton'] = F.to_tensor(sample['skeleton'])
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # normalise image
        sample['image'] = F.normalize(sample['image'], mean=self.mean, std=self.std)
        sample['skeleton'] = (sample['skeleton'] > 0).float()
        h, w = sample['skeleton'].shape[1:3]
        ih, iw = sample['image'].shape[1:3]
        assert (h, w) == (ih, iw)

        # normalise points
        for sk_key in sk_info:
            if sk_key in sample:
                sample[sk_key] = torch.clip(sample[sk_key], min=0.0, max=1.0)
        for per_branch in sample['branches']:
            for k in branch_info:
                per_branch[k] = torch.clip(per_branch[k], min=0.0, max=1.0)

        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

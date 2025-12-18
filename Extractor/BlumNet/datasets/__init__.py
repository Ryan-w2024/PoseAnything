from .skeleton import DATASET_OPTIONS
from .skeleton import build as build_skeleton


def build_dataset(image_set, args):
    if args.dataset_file in DATASET_OPTIONS:
        return build_skeleton(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')

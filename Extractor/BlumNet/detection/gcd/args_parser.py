import argparse
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=160, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='swin_small', type=str,
                        choices=['vgg16', 'resnet50', 'resnet101', 'inception_v3', 'swin_tiny', 'swin_small', 'swin_base'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1024, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=8, type=int)
    parser.add_argument('--enc_n_points', default=8, type=int)

    # * output
    parser.add_argument('--out_pts', type=int, default=0, help='output endpoints and junctions')
    parser.add_argument('--gid', action='store_true', help="output graph id")
    parser.add_argument('--aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--pts_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--rule', type=str, default='overlap_10_0.6')
    parser.add_argument('--dataset_file', type=str, default='sk1491',
                        choices=['SmithsonianLeaves', 'sk1491', 'sk1491_mix', 'sk1491_stitch', 'sk506', 'SYMMAX300', 'SymPASCAL',
                                 "animal2000", "kimia216",  "tetrapod120", "em200", "mpeg7",
                                 "SwedishLeaves", "WH-SYMMAX", "shapes-combine"])
    parser.add_argument('--datafile_mid', type=str, default='', choices=['', '_im', '_sp', '_imsp', '_flip', '_rot', '_scl'])
    parser.add_argument('--npt', type=int, default=2, choices=[2, 3, 5, 16])
    parser.add_argument('--data_root', default="./datasets", type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_score', default=0.65, type=float, help='threshold of prediction')
    parser.add_argument('--visual_type', default="lines", type=str, choices=['lines', 'lines&pts', "branches"], help='visualization in testing')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from .position_encoding import build_position_encoding
from .backbones.create_inception import Inception3
from .backbones.swin_transformer import SwinTransfomerNet
from .backbones.vgg import VGGfs, cfg
from BlumNet.lib.misc import NestedTensor, is_main_process


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, name='inception_v3'):
        super().__init__()
        if name in ['resnet50', 'resnet101']:
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)
            if return_interm_layers:
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
            else:
                return_layers = {'layer4': "0"}
                self.strides = [32]
                self.num_channels = [2048]
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        elif name == 'inception_v3':
            self.body = backbone # return keys = {"0", "1", "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [288, 768, 2048]
        elif name in ['swin_base', 'swin_tiny', 'swin_small',]:
            self.body = backbone # return keys = {"0", "1", "2"}
            self.strides = [8, 16, 32]
            dim = 128 if name == 'swin_base' else 96
            self.num_channels = [dim * 2, dim * 4, dim * 8]
        elif name == 'vgg16':
            self.body = backbone
            self.strides = [8, 16]
            self.num_channels = [512, 512]
        else:
            raise NotImplementedError('')
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        pretrained_swin_paths = {'swin_base':'pretrained/swin_base_patch4_window12_384_22k.pth',
                                 'swin_small': 'pretrained/swin_small_patch4_window7_224.pth',
                                 'swin_tiny': 'pretrained/swin_tiny_patch4_window7_224.pth'}
        pretrained = True
        if name == 'inception_v3':
            backbone = Inception3()
            if pretrained:
                state_dict = torchvision.models.inception_v3(pretrained=True).state_dict()
                backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})
        elif name in pretrained_swin_paths:
            pretrained_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", pretrained_swin_paths[name])
            backbone = SwinTransfomerNet(name)
            backbone.init_weights(pretrained=pretrained_path)
            print(f"Finished loading official pretraied weights: {pretrained_path}. "
                  "Please ignore above mismatch 'size mismatch for layers.x.blocks.x.attn.relative_position_index'")
        elif name in ['resnet50', 'resnet101']:
            norm_layer = FrozenBatchNorm2d
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=pretrained, norm_layer=norm_layer)
        elif name == 'vgg16':
            backbone = VGGfs(cfg['D'])
            pretrained_path = "pretrained/vgg16_caffe-292e1171.pth"
            state_dict = torch.load(pretrained_path)
            # state_dict = torchvision.models.vgg16(pretrained=True)
            backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})
            print("Finished loading official pretraied weights of vgg16. ")
        else:
            raise NotImplementedError('')
        super().__init__(backbone, train_backbone, return_interm_layers, name)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model

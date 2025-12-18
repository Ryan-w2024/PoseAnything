# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------


import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from .backbone import build_backbone
from .matcher import build_matcher
from .loss import AELoss, GraphIDLoss, sigmoid_focal_loss
from .deformable_transformer import build_deforamble_transformer
from BlumNet.lib.libcommon import SkPts
from BlumNet.lib.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

skparser = SkPts()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, cpts=1, gid=True, out_pts=0):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            cpts: int, number of points to describe a curve
            out_pts, int, the output number of end points and junction points
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2 * cpts, 3)
        self.num_feature_levels = num_feature_levels
        # for graph prediction
        self.gid = gid
        self.gid_embed = nn.Linear(hidden_dim, 1) if gid else None
        assert out_pts >= 0
        self.out_pts = out_pts
        pts_class = 3  # 0-endpts, 1-junctions, 2-nontarget
        if out_pts > 0:
            self.class_pt_embed = nn.Linear(hidden_dim, pts_class)
            self.pt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries + out_pts, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if out_pts > 0:
            self.class_pt_embed.bias.data = torch.ones(pts_class) * bias_value
            nn.init.constant_(self.pt_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.pt_embed.layers[-1].bias.data, 0)
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if out_pts > 0:
                self.class_pt_embed = nn.ModuleList([self.class_pt_embed for _ in range(num_pred)])
                self.pt_embed = nn.ModuleList([self.pt_embed for _ in range(num_pred)])
            if self.gid:
                self.gid_embed = nn.ModuleList([self.gid_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, inter_references, _, _ = self.transformer(srcs, masks, pos, self.query_embed.weight)
        pts_hs, pts_init_refer, pts_inter_refer = (
            hs[:, :, :self.out_pts], init_reference[:, :self.out_pts], inter_references[:, :, :self.out_pts])
        hs, init_reference, inter_references = (
            hs[:, :, self.out_pts:], init_reference[:, self.out_pts:], inter_references[:, :, self.out_pts:])

        rst = {}
        rst['curves'] = self._forward(
            hs, init_reference, inter_references, class_embed=self.class_embed, bbox_embed=self.bbox_embed)
        if self.out_pts > 0:
            rst['pts'] = self._forward(
                pts_hs, pts_init_refer, pts_inter_refer, class_embed=self.class_pt_embed, bbox_embed=self.pt_embed)

        return rst

    def _forward(self, hs, init_reference, inter_references, class_embed, bbox_embed, key_prefix=''):
        outputs_classes = []
        outputs_coords = []
        outputs_gids = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = class_embed[lvl](hs[lvl])
            outputs_gid = self.gid_embed[lvl](hs[lvl]) if self.gid else None
            tmp = bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                repeat_num = tmp.shape[-1] // reference.shape[-1]
                tmp = tmp + torch.cat([reference for i in range(repeat_num)], dim=-1)
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_gids.append(outputs_gid)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_gid = torch.stack(outputs_gids) if self.gid else None
        out = {f'{key_prefix}pred_logits': outputs_class[-1], f'{key_prefix}pred_boxes': outputs_coord[-1]}
        if self.gid:
            out[f'{key_prefix}pred_gids'] = outputs_gid[-1]
        if self.aux_loss:
            out[f'{key_prefix}aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_gid, key_prefix=key_prefix)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_gid=None, key_prefix=''):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_gid is None:
            return [{f'{key_prefix}pred_logits': a, f'{key_prefix}pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{f'{key_prefix}pred_logits': a, f'{key_prefix}pred_boxes': b, f'{key_prefix}pred_gids': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_gid[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses, gt_pts_key, gt_pts_label, focal_alpha=0.25, gid_label=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.gt_pts_key = gt_pts_key
        self.gt_pts_label = gt_pts_label
        self.gid_label = gid_label
        self.ae_loss_fun = GraphIDLoss() # AELoss()

    def loss_labels(self, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        non_tgt_label = src_logits.shape[-1] - 1
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[gt_pts_label][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], non_tgt_label, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t[gt_pts_key][i] for t, (_, i) in zip(targets, indices)], dim=0).flatten(1)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='mean') * 4
        losses = {'loss_bbox': loss_bbox}
        return losses

    def loss_gid_regress(self, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label):
        assert 'pred_gids' in outputs
        src_logits = outputs['pred_gids'].squeeze(dim=2)
        unmatched_gid = 0
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[gt_pts_label][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape, unmatched_gid, dtype=torch.float32, device=src_logits.device)
        target_classes[idx] = target_classes_o
        losses = {'loss_gid': 0}
        for _loss in self.ae_loss_fun(src_logits, target_classes):
            losses['loss_gid'] += torch.mean(_loss)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'gids': self.loss_gid_regress,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs)

    def forward(self, outputs, targets):
        loss = {}
        if 'curves' in outputs:
            gt_pts_key, gt_pts_label = self.gt_pts_key, self.gt_pts_label
            per_loss = self._forward(outputs['curves'], targets, gt_pts_key, gt_pts_label)
            loss.update(
                {f"c{k}": v for k, v in per_loss.items()}
            )
        if 'pts' in outputs:
            per_loss = self._forward(outputs['pts'], targets, gt_pts_key='key_pts', gt_pts_label='plabels')
            loss.update(
                {f"p{k}": v for k, v in per_loss.items()}
            )
        return loss

    def _forward(self, outputs, targets, gt_pts_key, gt_pts_label):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = outputs['pred_boxes'].device
        npt = outputs['pred_boxes'].shape[-1] // 2

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t[gt_pts_label]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}

        if npt > 5:
            pt_ids = torch.arange(npt, dtype=torch.long, device=device)
        else:
            pt_ids = torch.as_tensor([0, npt - 1], dtype=torch.long, device=device)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets, gt_pts_key, gt_pts_label, pt_ids=pt_ids)
        l_dict = {}
        for loss in self.losses:
            kwargs = {}
            l_dict.update(self.get_loss(
                loss, outputs_without_aux, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs))
        if self.gid_label is not None:
            l_dict.update(self.get_loss(
                'gids', outputs_without_aux, targets, indices, num_boxes, '', self.gid_label))
        losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, gt_pts_key, gt_pts_label, pt_ids=pt_ids)
                l_dict = {}
                for loss in self.losses:
                    kwargs = {} if (loss != 'labels') else {'log': True}
                    l_dict.update(self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs))
                if self.gid_label is not None:
                    l_dict.update(self.get_loss(
                        'gids', outputs, targets, indices, num_boxes, '', self.gid_label))
                losses.update({f"{k}_{i}": v for k, v in l_dict.items()})

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 2 #[target, non-target]

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        cpts=args.npt,
        gid=args.gid,
        out_pts=args.out_pts,
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_gid': 0.5,
    }
    add_items = {}
    for k, v in weight_dict.items():
        for prefix in ['c', 'p']:
            if prefix == 'c':
                add_items[f"{prefix}{k}"] = v
            else:
                relative_v = v / 10 # among all query, only about 10% is for points
                add_items[f"{prefix}{k}"] = args.pts_loss_coef * relative_v
    weight_dict.update(add_items)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        matcher, weight_dict, 
        losses=['labels', 'boxes'],
        gt_pts_key="curves",
        gt_pts_label='clabels',
        gid_label=("gids" if args.gid else None), # graph id is not used for most datasets
        focal_alpha=args.focal_alpha,
    )
    criterion.to(device)

    return model, criterion

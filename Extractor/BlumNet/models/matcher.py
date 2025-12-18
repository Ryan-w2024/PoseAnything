# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.wds = 1
        self.nonlinear_dist = False

        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    def set_wds(self, wds):
        assert wds >= 1
        self.wds = wds

    def forward(self, outputs, targets, gt_pts_key, gt_pts_label, focal_ce_cost=False, pt_ids=None):
        if self.wds == 1:
            return self.forward_old(outputs, targets, gt_pts_key, gt_pts_label, focal_ce_cost, pt_ids)
        else:
            return self.forward_windows(outputs, targets, gt_pts_key, gt_pts_label, focal_ce_cost)

    def forward_old(self, outputs, targets, gt_pts_key, gt_pts_label, focal_ce_cost=False, pt_ids=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_pts": Tensor of dim [batch_size, num_queries, 2] with the predicted skeleton coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "pts": Tensor of dim [num_target_boxes, 2] containing the target skeleton coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_pts = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 2]
            npt = out_pts.shape[-1] // 2
            pt_ids = torch.arange(npt, dtype=out_pts.device) if (pt_ids is None) else pt_ids
            out_pts = out_pts.view(-1, npt, 2)[:, pt_ids].flatten(1)

            # Also concat the target labels and boxes
            # tgt_ids = torch.cat([v["labels"] for v in targets])
            # tgt_pts = torch.cat([v["boxes"] for v in targets])
            tgt_ids = torch.cat([v[gt_pts_label] for v in targets])
            tgt_pts = torch.cat([v[gt_pts_key] for v in targets])
            tgt_pts = tgt_pts[:, pt_ids].flatten(1)
            # Compute the classification cost.
            if focal_ce_cost:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between points
            l1_cost = torch.cdist(out_pts, tgt_pts, p=1)
            if self.nonlinear_dist:
                # f(x) = x if (x <= t) else (a * (x**2) + b)
                # (t, a, b) = (0.1, 5,   0.05)
                # (t, a, b) = (0.2, 2.5, 0.1 )
                # (t, a, b) = (0.5, 1,   0.25)
                nonlinear_cost = torch.pow(l1_cost, 2) * 2.5 + 0.1
                l1_mask = (l1_cost <= 0.2).int()
                cost_fuse = l1_mask * l1_cost + (1 - l1_mask) * nonlinear_cost
                cost_bbox = 4 / tgt_pts.shape[-1] * cost_fuse
            else:
                cost_bbox = 4 / tgt_pts.shape[-1] * l1_cost

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()

            # sizes = [len(v["boxes"]) for v in targets]
            sizes = [len(v[gt_pts_key]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def forward_windows(self, outputs, targets, gt_pts_key, gt_pts_label, focal_ce_cost=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_pts": Tensor of dim [batch_size, num_queries, 2] with the predicted skeleton coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "pts": Tensor of dim [num_target_boxes, 2] containing the target skeleton coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert 'pred_boxes' in outputs
        assert 'pred_logits' in outputs
        src_pts = outputs['pred_boxes']
        src_logits = outputs['pred_logits']
        with torch.no_grad():
            out_prob = src_logits.sigmoid()
            src_cts = 0.5 * (src_pts[:, :, :2] + src_pts[:, :, 2:])
            tgt_cts = [torch.mean(t[gt_pts_key], dim=1) for t in targets]
            tgt_pts = [t[gt_pts_key].flatten(1) for t in targets]

            wds = self.wds
            indexes = []
            for batch_i in range(len(targets)):
                ind_is, ind_js = [], []
                for w_id in range(wds ** 2):
                    wr, wc = w_id // wds, w_id % wds
                    xmin, ymin = wc / wds, wr / wds
                    xmax, ymax = (wc + 1) / wds, (wr + 1) / wds
                    src_ids = (src_cts[batch_i][:, 0] >= xmin) * (src_cts[batch_i][:, 1] >= ymin) * \
                              (src_cts[batch_i][:, 0] < xmax) * (src_cts[batch_i][:, 1] < ymax)
                    tgt_ids = (tgt_cts[batch_i][:, 0] >= xmin) * (tgt_cts[batch_i][:, 1] >= ymin) * \
                              (tgt_cts[batch_i][:, 0] < xmax) * (tgt_cts[batch_i][:, 1] < ymax)
                    wd_src_pts = src_pts[batch_i][src_ids]
                    wd_tgt_pts = tgt_pts[batch_i][tgt_ids]
                    wd_prob = out_prob[batch_i][src_ids]
                    wd_tgt_labels = targets[batch_i][gt_pts_label][tgt_ids]
                    wd_cost = 4 / wd_tgt_pts.shape[-1] * torch.cdist(wd_src_pts, wd_tgt_pts, p=1)

                    # Compute the classification cost.
                    if focal_ce_cost:
                        alpha = 0.25
                        gamma = 2.0
                        neg_cost_class = (1 - alpha) * (wd_prob ** gamma) * (-(1 - wd_prob + 1e-8).log())
                        pos_cost_class = alpha * ((1 - wd_prob) ** gamma) * (-(wd_prob + 1e-8).log())
                        cost_class = pos_cost_class[:, wd_tgt_labels] - neg_cost_class[:, wd_tgt_labels]
                    else:
                        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                        # but approximate it in 1 - proba[target class].
                        # The 1 is a constant that doesn't change the matching, it can be ommitted.
                        cost_class = -wd_prob[:, wd_tgt_labels]

                    C = 5 * wd_cost + cost_class
                    ind_i, ind_j = linear_sum_assignment(C.cpu())
                    ind_i = torch.where(src_ids)[0][ind_i]
                    ind_j = torch.where(tgt_ids)[0][ind_j]
                    ind_is.append(ind_i)
                    ind_js.append(ind_j)

                ind_i, ind_j = torch.cat(ind_is), torch.cat(ind_js)
                indexes.append((ind_i, ind_j))
            return indexes


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox)

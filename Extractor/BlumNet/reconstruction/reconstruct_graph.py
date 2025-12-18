# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch import nn
from skimage import morphology
from scipy.optimize import linear_sum_assignment
from BlumNet.preparation import split_skeleton, connect_edges
from BlumNet.preparation import END_PT, JUNCTION_PT
from BlumNet.lib.hist_ftc_seg import ftc_seg


class GetElements(nn.Module):
    def __init__(self, eval_score=None, eval_pt_score=0.05):
        super(GetElements, self).__init__()
        self.eval_score = eval_score
        self.eval_pt_score = eval_pt_score

    def forward(self, outputs, target_sizes, non_target_label=None, from_curves=True):
        if from_curves:
            return self.forward_from_curves(outputs, target_sizes, non_target_label)
        else:
            return self.forward_from_pts(outputs, target_sizes)

    def forward_from_pts(self, outputs, target_sizes):
        """This forward is to parse the point outputs.
        
        Args:
            outputs: the outputs are produced at DeformableDETR inference, and in 
                training the outputs are supervised by the grountruth juntion/end points. 
            target_sizes: the input images' size 
        
        Return:
            (tgt_scores, out_labels, pred_boxes, out_pts), the first three items are tensors 
                    of points prediction from DeformableDETR, the last item is the parsed 
                    junction/end points.
        """

        def nms_pts(pts, scores, gids, dist_thresh=5):
            """NMS on the predicted points
            Args:
                pts: tensor of shape (num, 2), pixel scale
                scores: tensor of shape (num,), float from 0. to 1.
                gids: tensor of shape (num,), float.
            Return:
                (npts, nscores), same format like the input.
            """
            index = torch.argsort(scores, dim=0, descending=True)
            scores, pts = scores[index], pts[index]
            len_pts = len(pts)
            dist_m = torch.cdist(pts, pts, p=2) + 10 * torch.eye(n=len_pts, device=pts.device)
            retained = torch.ones(size=(len_pts,), dtype=torch.int32, device=pts.device)
            is_suppressed = dist_m < dist_thresh
            for i in range(len_pts):
                if retained[i] < 1:
                    continue
                for _nms_id in torch.where(is_suppressed[i])[0].cpu().numpy().tolist():
                    if _nms_id > i:
                        retained[_nms_id] = 0
            retained = retained > 0
            nscores, npts, gids = scores[retained], pts[retained], gids[retained]
            return nscores, npts, gids

        out_logits, out_boxes = outputs['pred_logits'], outputs['pred_boxes']
        assert out_logits.shape[0] == 1, 'batchsize != 1'
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        out_gid = outputs.get(
            'pred_gids', torch.ones_like(out_logits[..., 0:1]))
        out_gids = out_gid.squeeze(dim=2)

        out_prob = F.softmax(out_logits, -1)  # (bs, num_query, num_class)
        out_scores, out_labels = out_prob.max(-1)
        tgt_scores = 1 - out_prob[..., -1]
        endpt_scores = out_prob[..., END_PT - 1]
        junpt_scores = out_prob[..., JUNCTION_PT - 1]

        out_pts, pred_boxes = [], []
        for endpt_score, junpt_score, gids, pts, im_wh in zip(endpt_scores, junpt_scores, out_gids, out_boxes, target_sizes):
            pd_pts_bsize = pts * im_wh
            pred_boxes.append(pd_pts_bsize)
            np_pts = pd_pts_bsize
            # filter
            endpt_ids = (endpt_score >= self.eval_pt_score)
            junct_ids = (junpt_score >= self.eval_pt_score)

            endpt_score, endpts, endpt_gids = nms_pts(np_pts[endpt_ids], endpt_score[endpt_ids], gids[endpt_ids], dist_thresh=5)
            junpt_score, juncpts, junpt_gids = nms_pts(np_pts[junct_ids], junpt_score[junct_ids], gids[junct_ids], dist_thresh=5)

            out_pts.append((
                (np.round(endpts.cpu().detach().numpy()), np.round(juncpts.cpu().detach().numpy())),
                (endpt_score.cpu().detach().numpy(), junpt_score.cpu().detach().numpy()),
                (endpt_gids.cpu().detach().numpy(), junpt_gids.cpu().detach().numpy())
            ))
        pred_boxes = torch.stack(pred_boxes, dim=0)
        return tgt_scores, out_labels, pred_boxes, out_pts

    def forward_from_curves(self, outputs, target_sizes, non_target_label):
        """This forward is to parse the lines/curves and get branches by projection on mask.
        
        Args:
            outputs: the outputs are produced at DeformableDETR inference, and in 
                training the outputs are supervised by the grountruth lines/curves. 
            target_sizes: the input images' size 
            non_target_label: int, the label of background.
        
        Return:
            (tgt_scores, out_labels, pred_boxes, out_sequences), the first three items are tensors 
                of curves/lines prediction from DeformableDETR, the last item is the parsed branches.
        """
        out_logits, out_lines = outputs['pred_logits'], outputs['pred_boxes']
        assert out_logits.shape[0] == 1, 'batchsize != 1'
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        out_gid = outputs.get(
            'pred_gids', torch.ones_like(out_logits[..., 0:1]))
        out_gids = out_gid.squeeze(dim=2)

        out_prob = F.softmax(out_logits, -1)       # (bs, num_query, num_class)
        out_scores, out_labels = out_prob.max(-1)  # prob[..., :-1].max(-1)
        tgt_scores = 1 - out_prob[..., -1]
        out_sequences = []
        pred_boxes = []
        _npt = out_lines.shape[-1] // 2
        for labels, lines, gids, im_wh, scores in zip(out_labels, out_lines, out_gids, target_sizes, tgt_scores):
            pd_lines = lines.view(-1, _npt, 2)
            pd_lines_bsize = pd_lines * im_wh
            pred_boxes.append(pd_lines_bsize.flatten(1))
            labels = labels.data.cpu().numpy().astype(np.int32)
            np_lines = np.round(pd_lines_bsize.cpu().detach().numpy())
            w, h = im_wh.data.cpu().int()
            # filter
            if self.eval_score is None:
                ids = (labels != non_target_label)
            else:
                ids = scores.cpu().detach().numpy() >= self.eval_score
            indexes = torch.arange(lines.shape[0], device=im_wh.device, dtype=torch.long)
            pd_pts, labels, gids = np_lines[ids].astype(np.int32), labels[ids], gids[ids]
            pd_lines_bsize = pd_lines_bsize[ids]
            pd_indexes = indexes[ids]

            pred_mask = np.zeros((h, w), dtype=np.uint8)
            pd_num = pd_pts.shape[0]
            if _npt == 1:
                pd_pts = pd_pts.reshape((-1, 2))
                pred_mask[pd_pts[:, 1], pd_pts[:, 0]] = 255
            elif _npt > 1:
                cv2.polylines(pred_mask, pd_pts, False, color=255, thickness=1)
            else:
                raise ValueError('')
            dks = 5
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks))
            pred_mask = cv2.dilate(pred_mask, element) > 0
            # thinning to the width of 1-pixel
            pred_mask = morphology.skeletonize(
                (morphology.skeletonize(pred_mask, method='lee') > 0).astype(np.uint8)
            ).astype(np.uint8)
            # projection to branches
            branches = split_skeleton(pred_mask)
            branches = [b for b in branches if len(b) > 2]
            branches = connect_edges(branches, dist_thresh=3)
            branches = [b for b in branches if len(b) > 5]
            branches = connect_edges(branches, dist_thresh=6)

            b_num = len(branches)
            if b_num == 0:
                out_sequences.append([])
                continue

            # make sure enough branch points to match the predicted lines
            branches_len, branches_cat = [0], []
            for b in branches:
                branches_len.append(len(b))
                branches_cat.extend(b)
            pts_num = sum(branches_len)

            if pts_num < (4 * pd_num):
                scl = 4 * pd_num / pts_num
                branches_len, branches_cat = [0], []
                for b in branches:
                    len_b = int(len(b) * scl)
                    b = F.interpolate(
                        torch.as_tensor(b, dtype=torch.float32)[None, None, ...],
                        size=(len_b, 2), mode='bilinear', align_corners=True)[0, 0]
                    branches_len.append(len_b)
                    branches_cat.append(b)
            pts_num = sum(branches_len)

            # get the branch sequence for each branch

            mapping_dis = 0
            if isinstance(branches_cat[0], torch.Tensor):
                branches_cat = torch.cat(branches_cat, dim=0).to(pd_lines)
            else:
                branches_cat = torch.as_tensor(branches_cat).to(pd_lines)
            tmp_branches = branches_cat[:, None, :].expand(pts_num, pd_num, 2)
            for pd_lines_bpts in [pd_lines_bsize[:, 0], pd_lines_bsize[:, -1]]:
                mapping_diff = tmp_branches - pd_lines_bpts[None, :, :].expand(pts_num, pd_num, 2)
                mapping_dis = mapping_dis + torch.norm(mapping_diff, p=2, dim=-1)

            ind_i, ind_j = linear_sum_assignment(mapping_dis.cpu().detach().numpy())
            ind_i, ind_j = torch.as_tensor(ind_i), torch.as_tensor(ind_j)

            per_sequences = []
            pts_cumsum = np.cumsum(branches_len).tolist()
            for bid, id_floor, id_ceil in zip(range(b_num), pts_cumsum[:-1], pts_cumsum[1:]):
                i_per_branch = torch.logical_and(ind_i < id_ceil, ind_i >= id_floor)
                j_per_branch = ind_j[i_per_branch]
                if j_per_branch.shape[0] == 0:
                    continue
                branch_sq = pd_indexes[j_per_branch] # pd_lines_bsize[j_per_branch]
                branch_gid = torch.mean(gids[j_per_branch]).cpu().detach().numpy().item()
                per_sequences.append((branch_sq, branches[bid], branch_gid))
            out_sequences.append(per_sequences)
        pred_boxes = torch.stack(pred_boxes, dim=0)
        return tgt_scores, out_labels, pred_boxes, out_sequences


class PostProcess(nn.Module):
    def __init__(self, eval_score):
        super(PostProcess, self).__init__()
        self.forward_elements = GetElements(eval_score)

    def forward(self, outputs, target_sizes, ignore_graph=False):
        results = {}
        for k in outputs.keys():
            if k == 'curves':
                if ignore_graph:
                    results[k] = self._forward_nongraph(outputs[k], target_sizes)
                else:
                    results[k] = self._forward_curves(outputs[k], target_sizes)
            elif k == 'pts':
                results[k] = self._forward_pts(outputs[k], target_sizes)
            else:
                raise NotImplementedError('')
        if (not ignore_graph) and ('pts' in results) and ('curves' in results):
            results['graphs'] = self.reconstruct_graphs(
                results['curves'], results['pts'], target_sizes, ignore_gid=True) # For AELoss set ignore_gid=False; for GraphIDLoss set ignore_gid=True

        return results
    
    @staticmethod
    def visualise_curves(pred, eval_score, src_img, dks=3, thinning=False, ch3mask=False, vmask=1):
        """Visualise the predicted curves with postprocessing.
        Tips: for SymPASCAL dks=5, for other datasets, dks=3

        Args:
            pred, the result of ${self._forward_curves}[0]
            eval_score, float, score threshold of curves
            src_img, array of input image

        Return:
            img, array of drawing predictions on src_img
            pred_mask, array of drawing predictions on mask
        """
        h, w = src_img.shape[:2]
        pred_mask = np.zeros((h, w), dtype=np.uint8)

        npt = pred['lines'].shape[-1] // 2
        assert npt > 1

        _scores = pred['scores'].data.cpu().numpy()
        _lines = pred['lines'].view(-1, npt, 2).data.cpu().numpy().astype(np.int32)
        ids = _scores > eval_score
        pred_pts = _lines[ids].astype(np.int32)

        cv2.polylines(pred_mask, pred_pts, False, color=255, thickness=1) # for SymPASCAL thickness=3
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks))
        pred_mask = cv2.dilate(pred_mask, element)
        if thinning: # fast thinning
            pred_mask = morphology.skeletonize(pred_mask, method='lee').astype(np.uint8) 
            pred_mask = cv2.dilate(pred_mask, element)
        pred_mask_3ch = (cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR) > 0)
        pred_mask = pred_mask_3ch[:, :, 0]
        img = np.copy(src_img)
        img = img * (1 - pred_mask_3ch)
        img[:, :, 2] = img[:, :, 2] + 255 * pred_mask   # R
        img[:, :, 1] = img[:, :, 1] + 255 * pred_mask   # G
        img[:, :, 0] = img[:, :, 0] + 255 * pred_mask   # B
        if ch3mask:
            return img, pred_mask_3ch * vmask
        return img, pred_mask * vmask

    @staticmethod
    def visualise_pts(pred, eval_score, src_img):
        """Visualise the predicted curves with postprocessing.
        Tips: for SymPASCAL dks=5, for other datasets, dks=3

        Args:
            pred, the result of ${self._forward_pts}[0]
            eval_score, float, score threshold of points
            src_img, array of input image

        Return:
            img, array of drawing predictions on src_img
            pred_mask, array of drawing predictions on mask
        """
        JUNCTION_PT_COLOR = (0, 0, 255)
        END_PT_COLOR = (0, 255, 0)
        h, w, _ = src_img.shape
        pred_mask = np.zeros((h, w), dtype=np.uint8)

        (endpts, juncpts), (endpt_score, junpt_score), _ = pred['pts']
        endpts, juncpts = endpts.astype(np.int32), juncpts.astype(np.int32)
        endpts = endpts[endpt_score > eval_score].tolist()
        juncpts = juncpts[junpt_score > eval_score].tolist()
        img = np.copy(src_img).astype(np.uint8)
        for i in range(len(endpts)):
            cv2.circle(img, tuple(endpts[i]), radius=2, color=END_PT_COLOR, thickness=2)
            cv2.circle(pred_mask, tuple(endpts[i]), radius=2, color=255, thickness=2)
        for i in range(len(juncpts)):
            cv2.circle(img, tuple(juncpts[i]), radius=2, color=JUNCTION_PT_COLOR, thickness=2)
            cv2.circle(pred_mask, tuple(juncpts[i]), radius=2, color=128, thickness=2)

        return img, pred_mask

    @torch.no_grad()
    def _forward_nongraph(self, outputs, target_sizes):
        """ Fast forward without graph information / graph id, only the curves/lines prediction and 
        points prediction is processed. This forward is recommended for most usages since it covers 
        the rough outline of the predicted skeleton and time cost is much lower than forward with 
        graph reconstruction.

        Args:
            outputs: the outputs are produced at DeformableDETR inference, and in training the outputs 
                    are supervised by the grountruth lines/curves. 
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each image.
        
        Return:
            results, list of dict, each dict is the predicted curves
        """
        out_logits, out_line = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1) # prob[..., :-1].max(-1)
        scores = 1 - prob[..., -1]    # the target score

        # convert to [x0, y0, x1, y1] format
        img_w, img_h = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h] * (out_line.shape[-1] // 2), dim=1)
        lines = out_line * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'lines': b} for s, l, b in zip(scores, labels, lines)]

        return results

    @torch.no_grad()
    def _forward_pts(self, outputs, target_sizes):
        """ Get the detail info of junction/end points and the parsed(NMS) output.
 
        Args:
            outputs: the outputs are produced at DeformableDETR inference, and in 
                training the outputs are supervised by the grountruth juntion/end points. 
            target_sizes: the input images' size 
        
        Return:
            results, list of dict, each dict is the predicted details of points
        """
        out_scores, out_labels, out_lines, out_pts = self.forward_elements(outputs, target_sizes, from_curves=False)
        out_gid = outputs.get('pred_gids', None)
        if out_gid is not None:
            out_gids = out_gid.squeeze(dim=2)
            results = [{'scores': s, 'labels': l, 'lines': b, 'gids': t, 'pts': pts}
                       for s, l, b, t, pts in
                       zip(out_scores, out_labels, out_lines, out_gids, out_pts)]
        else:
            results = [{'scores': s, 'labels': l, 'lines': b, 'pts': pts}
                       for s, l, b, pts in
                       zip(out_scores, out_labels, out_lines, out_pts)]

        return results

    @torch.no_grad()
    def _forward_curves(self, outputs, target_sizes):
        """ Get the detail info of lines/curves and the parsed branches.
 
        Args:
            outputs: the outputs are produced at DeformableDETR inference, and in 
                training the outputs are supervised by the grountruth lines/curves. 
            target_sizes: the input images' size 
        
        Return:
            results, list of dict, each dict is the predicted details of lines and branches.
        """
        non_tgt_label = outputs['pred_logits'].shape[-1] - 1
        out_scores, out_labels, out_lines, out_sequences = self.forward_elements(
            outputs, target_sizes, non_tgt_label, from_curves=True)
        out_gid = outputs.get('pred_gids', None)

        if out_gid is not None:
            out_gids = out_gid.squeeze(dim=2)
            results = [{'scores': s, 'labels': l, 'lines': b, 'gids': t, 'sequences': sq, }
                       for s, l, b, t, sq in
                       zip(out_scores, out_labels, out_lines, out_gids, out_sequences)]
        else:
            results = [{'scores': s, 'labels': l, 'lines': b, 'sequences': sq}
                       for s, l, b, sq in
                       zip(out_scores, out_labels, out_lines, out_sequences)]

        return results

    @staticmethod
    @torch.no_grad()
    def reconstruct_graphs(rst_curves, rst_pts, target_sizes, gama=0.05, ignore_gid=False):
        """ Reconstruct graphs from the predicted branches and points. 

        Args:
            rst_curves: list, its size is the batchsize, each item is a dict like 
                       {'sequences': (curve_indexes_array, branch_points_array, graph_id_float), 
                        'scores': s_tensor, 'labels': l_tensor, 'lines': lines_tensor, 'gids': gids_tensor}
            rst_pts: list, its size is the batchsize, each item is a dict like 
                       {'pts': ((endpts_array, juncpts_array), 
                                (endpt_score_array, junpt_score_array), 
                                (endpt_gids_array, junpt_gids_array)),
                        }
            target_sizes: tensor or list, each item is (w, h)
            gama: float, threshold to filter unqualified branch-point pairs
            ignore_gid: bool, set true if the input image definitely has only one skeleton graph.

        Return:
            rst_graphs, list, its size is the batchsize, each item is a dict like {'graph_id': (branches, points)}
        """
        def perpendicular_distance(branches, pts):
            """Get the dilation distance among M branches and N pts.
            Return:
                dilation_matrix, array of shape (M, N)
            """
            branches = [np.float32(b) for b in branches]
            pts = np.float32(pts)
            ptslen = pts.shape[0]
            dilation_matrix = []
            for branch in branches:
                blen = branch.shape[0]
                deta_xy = np.repeat(branch[None, ...], ptslen, axis=0) - \
                          np.repeat(pts[:, None, :], blen, axis=1)
                z = deta_xy[..., 0] + deta_xy[..., 1] * 1j
                distances = np.abs(z)
                dilation_distance = np.min(distances, axis=1)
                dilation_matrix.append(dilation_distance)
            dilation_matrix = np.stack(dilation_matrix, axis=0)
            return dilation_matrix
        def pts_distance(pts1, pts2):
            """Get the dilation distance among M pts-pts1 and N pts-pts2.
            Return:
                distances, array of shape (M, N)
            """
            pts1 = np.float32(pts1)
            pts2 = np.float32(pts2)
            l1, l2 = pts1.shape[0], pts2.shape[0]
            deta_xy = np.repeat(pts1[:, None, ...], l2, axis=1) - \
                      np.repeat(pts2[None, ...], l1, axis=0)
            distances = np.linalg.norm(deta_xy, ord=2, axis=2)
            return distances

        SMALL_OFFSET = 1e-6
        rst_graphs = []
        batchsize = len(rst_curves)
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.cpu().numpy().tolist()  

        for _b in range(batchsize):
            # get graphs for each 
            branches = rst_curves[_b]['sequences'] # [(seq, branch, branch_gid), ... ]
            cgids = rst_curves[_b].get('gids', None)
            (endpts, juncpts), _, (endpt_gids, junpt_gids) = rst_pts[_b]['pts']
            ptgids = endpt_gids.tolist() + junpt_gids.tolist()
            pts = endpts.tolist() + juncpts.tolist()
            w, h = target_sizes[_b]
            if cgids is not None and not ignore_gid:
                # step 1 histogram segment on graph id
                per_gids = cgids.cpu().tolist() + ptgids
                gmax, gmin = np.max(per_gids), np.min(per_gids)
                bins = gmin + np.arange(65) / 64. * (gmax - gmin)
                hist, bin_edges = np.histogram(per_gids, bins=bins)
                bin_edges = bin_edges.tolist()
                idx_segs = ftc_seg(hist, e=1.0)
                graph_edges = bin_edges[:1] + \
                              [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in idx_segs] + \
                              bin_edges[-1:]

                # step2 update graph id for branches and points
                graph_num = len(graph_edges) - 1
                nbranches = []
                for (_, branch, branch_gid) in branches:
                    pd_gid = -1
                    for g_id in range(graph_num):
                        if branch_gid < graph_edges[1 + g_id] + SMALL_OFFSET:
                            pd_gid = g_id
                            break
                    if pd_gid == -1:
                        raise ValueError('Wrong graph id')
                    nbranches.append((_, branch, pd_gid))
                branches = nbranches
                nptgids = []
                for pgid in ptgids:
                    pd_gid = -1
                    for g_id in range(graph_num):
                        if pgid < graph_edges[1 + g_id] + SMALL_OFFSET:
                            pd_gid = g_id
                            break
                    if pd_gid == -1:
                        raise ValueError('Wrong graph id')
                    nptgids.append(pd_gid)
                ptgids = nptgids

            # step3 dilation and checking (dilation is to get the perpendicular distance,
            # an equivalent method of dilation is to find the nearest neighbors between
            # the branch points and the end/junction points.
            gid_set = set()
            gbranches = defaultdict(list)
            for (_, branch, pd_gid) in branches:
                if ignore_gid:
                    pd_gid = 1
                gbranches[pd_gid].append(branch)
                gid_set.add(pd_gid)
            gpts = defaultdict(list)
            for g, p in zip(ptgids, pts):
                if ignore_gid:
                    g = 1
                gpts[g].append(p)
                gid_set.add(g)
            graphs = dict()
            R = dict()
            for g in gid_set:
                if len(gbranches[g]) == 0 or len(gpts[g]) == 0:
                    continue
                dilation_matrix = perpendicular_distance(gbranches[g], gpts[g])
                p1s, p2s = [b[0] for b in gbranches[g]], [b[1] for b in gbranches[g]]
                d1 = pts_distance(p1s, gpts[g]) + dilation_matrix
                d2 = pts_distance(p2s, gpts[g]) + dilation_matrix
                index1, index2 = np.argmin(d1, axis=1), np.argmin(d2, axis=1)
                min_dis1, min_dis2 = np.min(d1, axis=1), np.min(d2, axis=1)
                #R[g] = (index1, min_dis1, index2, min_dis2)

                # step4 remove unqualified pairs
                qualified_d1 = min_dis1 < gama * ((w**2 + h**2)**0.5)
                qualified_d2 = min_dis2 < gama * ((w**2 + h**2)**0.5)
                qualified_brs = np.logical_or(qualified_d1, qualified_d2) # weakly remove some noises
                index1 = index1[qualified_brs]
                index2 = index2[qualified_brs]
                qualified_pts = [gpts[g][ii] for ii in set(index1.tolist() + index2.tolist())]
                num_brs = len(gbranches[g])
                qualified_bs = [gbranches[g][ii] for ii in range(num_brs) if qualified_brs[ii]]
                graphs[g] = (qualified_bs, qualified_pts)
                #print("get a graph")
            # get graphs
            rst_graphs.append(graphs)
        return rst_graphs


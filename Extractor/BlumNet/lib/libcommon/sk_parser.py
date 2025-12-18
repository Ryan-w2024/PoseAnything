# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
from scipy import signal
import skimage.draw as drawer
from skimage import morphology
from collections import defaultdict
from torch.functional import F


EIGHT_NBRS = [(-1, -1), (0, -1), (1, -1),
              (-1, 0),           (1, 0),
              (-1, 1),  (0, 1),  (1, 1)]
VH4_NBRS = [      (0, -1),
            (-1, 0),     (1, 0),
                  (0,  1),      ]
X4_NBRS = [(-1, -1),  (1, -1),
           (-1,  1),  (1,  1)]


H_TYPE = 0
V_TYPE = 1
OTHERS = -1


class HeatmapParser:
    """The heatmap parser for post processing."""

    def __init__(self, max_pool=(3, 1, 1), gauss_k=7, sigma=2.0):
        """
        Args:
           max_pool, (kernel, stride, pad)
           gauss_k, int
           sigma, float
        """
        self.pool = torch.nn.MaxPool2d(*max_pool)
        self.tensor_gauss = transforms.GaussianBlur(gauss_k, sigma=sigma)

    def gaussian(self, heatmaps):
        with torch.no_grad():
            heatmaps = self.tensor_gauss(heatmaps)
            return heatmaps

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """
        with torch.no_grad():
            maxm = self.pool(heatmaps)
            maxm = torch.eq(maxm, heatmaps).float()
            heatmaps = heatmaps * maxm
            return heatmaps

    @staticmethod
    def adjust(ans, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            ans (list(np.ndarray)): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """
        _, _, H, W = heatmaps.shape
        for batch_id, people in enumerate(ans):
            for people_id, people_i in enumerate(people):
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        x, y = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = heatmaps[batch_id][joint_id]
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1),
                                                             xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy,
                                                             max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id,
                                      0:2] = (x + 0.5, y + 0.5)
        return ans

    @staticmethod
    def topK(heatmap_pts, threshold, k=512, distance=0.05):
        """From a heatmap of shape h * w, get points [(x, y), ... ]
        which are top k scores and far away from each other"""
        pts, pts_score = [], []
        h, w = heatmap_pts.shape[:2]
        yy, xx = np.where(heatmap_pts > threshold)
        if len(yy) < 1:
            return pts, pts_score

        scores = heatmap_pts[yy, xx]
        index = np.argsort(scores)[::-1][:k]
        scores = scores[index]
        yy, xx = yy[index], xx[index]
        len_pts = len(yy)
        places = np.concatenate([xx[:,None], yy[:,None]], axis=1)
        dis_matrix = np.repeat(places[None,...], len_pts, axis=0) - np.repeat(places[:,None,...], len_pts, axis=1)
        dis_threshold = distance * ((h**2 + w**2)**0.5)
        l2_dis = np.linalg.norm(dis_matrix, axis=2) < max(dis_threshold, 5)
        ignore_ids = set()

        for i in range(len_pts):
            if i in ignore_ids:
                continue
            pts.append(places[i].tolist())
            pts_score.append(scores[i].item())
            ignore_ids = ignore_ids.union(set(np.where(l2_dis[i])[0].tolist()))
            if len(ignore_ids) == len_pts:
                break
        return pts, pts_score

def pol2cart(r, angle):
    '''
    Parameters:
    - r: float, vector amplitude
    - theta: float, vector angle in degrees
    Returns:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    '''
    theta = angle / 180.0 * np.pi
    z = r * np.exp(1j * theta)
    x, y = z.real, z.imag

    return x, y

def cart2pol(x, y):
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''

    z = x + y * 1j
    r, theta = np.abs(z), np.angle(z, True)

    return r, theta


class BaseParser:

    def get_patches(self, pts_set):
        """Given a set of points in a mask, find clusters of all points
        Args:
            pts_set, {(x, y), ... }

        Return:
            clusters, [cluster, ...], each cluster is [(x, y), ... ]
        """
        clusters = []
        while len(pts_set) > 0:
            x, y = pts_set.pop()
            patch = [(x, y)]
            nbr_pts = set(patch)
            while len(nbr_pts) > 0:
                (x, y) = nbr_pts.pop()
                new_nbr_pts = set(
                    [(x-dx, y-dy) for (dx, dy) in EIGHT_NBRS]
                ) - set(patch)
                new_nbr_pts = new_nbr_pts.intersection(pts_set)
                if len(new_nbr_pts) == 0:
                    continue
                pts_set = pts_set - new_nbr_pts
                for new_pt in new_nbr_pts:
                    patch.append(new_pt)
                nbr_pts = nbr_pts.union(new_nbr_pts)
            clusters.append(patch)
        return clusters

    def get_points_of_ptmask(self, pt_mask):
        """Get all points of given mask, mapping points to raw size.
        Args:
            pt_mask, 0-1 np.array

        Return:
            mpts, [(x, y), ... ]
        """
        mh, mw = pt_mask.shape[:2]
        place = np.argwhere(pt_mask > 0) # get location at (h, w)
        pts = [tuple(i) for i in place.tolist()]
        clusters = self.get_patches(set(pts))
        mpts = []
        for clt in clusters:
            clt = np.int32(clt)
            mean_yx = np.mean(clt, axis=0)
            # cy, cx = np.round(mean_yx).astype(np.int32).tolist()
            cy, cx = np.round(mean_yx).tolist()
            mpts.append((cx, cy))
        return mpts

    def get_lines_of_lnmask(self, lnmask, mpts, raw_scale=1.0, **kwargs):
        """
        Args:
            lnmask, H*W, 0-1 np.array
            mpts, [(x, y), ... ]
            mpts_rawpts, {(x, y):(raw_x, raw_y), ... }
            pol_masks, angle_num * r * r, 0-1 np.array
        """
        raise NotImplementedError('ABC function')

    def project_wd(self, lines):
        """ Project all doors, windows to walls.

        Args:
            all_lines, [wall, window, door], each item is [line, ... ],
                       each line is [(x1, y1), (x2, y2)]
        Return:
            new_lines, [wall, window, door]
        """
        # condition 1, the center of d-w is close enough to wall.
        # condition 2, the intersection has enough length.
        raise NotImplementedError('')

    def infer(self, masks, raw_scale=1.0, draw_line=True):
        """
        Args:
            masks, 6*H*W, the predicted heatmaps of skeletons and points. At dim=0,
                the order is wall_pt, window_pt, door_pt, wall_line, window_line, door_line.
            raw_h, int, original input size
            raw_w, int, original input size
        Return:
            all_lines, [wall, window, door], each item is [line, ... ],
                       each line is [(x1, y1), (x2, y2)]
            (window_2_wall_id, door_2_wall_id), mapping each window-door to a wall index.
            vismasks, None or array.
        """
        all_lines = []
        vismasks = []
        for ii in range(3):
            mpts = self.get_points_of_ptmask(
                masks[ii])
            lines, vismask = self.get_lines_of_lnmask(
                masks[ii + 3], mpts, raw_scale=raw_scale, draw_line=draw_line, is_wall=(ii==0))
            all_lines.append(lines)
            vismasks.append(vismask)
        all_lines, window_2_wall_id, door_2_wall_id = self.project_wd(all_lines)
        return (all_lines, window_2_wall_id, door_2_wall_id, vismasks)


class SkeletonParser(BaseParser):

    def __init__(self, edge_max_dis=4):
        self.edge_max_dis = edge_max_dis

    def project_wd(self, lines):
        wall, window, door = lines
        wd = window + door

        window_2_wall_id = []
        door_2_wall_id = []
        if len(wd) < 1:
            return lines, window_2_wall_id, door_2_wall_id
        points_0 = np.float32([line[0] for line in wd])
        points_1 = np.float32([line[1] for line in wd])

        new_window = []
        new_door = []
        for ii, line in enumerate(wall):
            (x1, y1), (x2, y2) = line[0], line[-1]
            det_x, det_y = abs(x2 - x1), abs(y2 - y1)
            if det_x + det_y < 5: # too short to have a door/window
                continue

            # condition 1, the center of door-window is close enough to wall.
            distances = self.distance(points_0, line[0], line[1]) + \
                        self.distance(points_1, line[0], line[1])
            condition1 = (distances < 10) #.astype(np.int32)
            if np.sum(condition1) < 1:
                continue
            # condition 2, the intersection has enough length.
            short_pts_0 = points_0[condition1]
            short_pts_1 = points_1[condition1]
            projects_0 = SkeletonParser.project_points(short_pts_0, line[0], line[1])
            projects_1 = SkeletonParser.project_points(short_pts_1, line[0], line[1])
            projects_0 = np.float32(projects_0)
            projects_1 = np.float32(projects_1)
            proj_len = np.linalg.norm(projects_1 - projects_0, axis=1)

            if det_x < det_y: # close to Y axis
                l1, r1 = min(y1, y2), max(y1, y2)
                tan_theta = np.tan(det_x / det_y)
                dim_id = 1
            else: # close to X axis
                l1, r1 = min(x1, x2), max(x1, x2)
                tan_theta = np.tan(det_y / det_x)
                dim_id = 0
            cos_theta = np.cos(np.arctan(tan_theta))
            less_mark = (projects_0[:, dim_id] < projects_1[:, dim_id]).astype(np.float32)
            l2 = less_mark * projects_0[:, dim_id] + (1 - less_mark) * projects_1[:, dim_id]
            r2 = (1 - less_mark) * projects_0[:, dim_id] + less_mark * projects_1[:, dim_id]
            common_r = r1 * (r2 > r1) + r2 * (r2 <= r1)
            common_l = l2 * (l2 > l1) + l1 * (l2 <= l1)
            det_common =  common_r - common_l
            intersect = det_common * (common_l < common_r) / cos_theta
            condition2 = (intersect > proj_len / 3) * (intersect > 10)

            select1_wd = np.where(condition1)[0][condition2]
            select2_wd = np.where(condition2)[0]
            if len(select2_wd) < 1:
                continue
            else:
                # Get a window-door for this wall
                if det_x < det_y:
                    _, lpts = self.get_axisloc((x1, y1), (x2, y2), yaxis=common_l[select2_wd])
                    _, rpts = self.get_axisloc((x1, y1), (x2, y2), yaxis=common_r[select2_wd])
                else:
                    lpts, _ = self.get_axisloc((x1, y1), (x2, y2), xaxis=common_l[select2_wd])
                    rpts, _ = self.get_axisloc((x1, y1), (x2, y2), xaxis=common_r[select2_wd])
                # avoid collection between two doors/windows
                fl_pts, fr_pts = np.float32(lpts), np.float32(rpts)
                lpts = fl_pts * 0.95 + fr_pts * 0.05
                rpts = fl_pts * 0.05 + fr_pts * 0.95

                window_index = select1_wd < len(window)
                door_index = select1_wd >= len(window)
                for index, new_wd, id_buffer in zip(
                    [window_index, door_index],
                    [new_window, new_door],
                    [window_2_wall_id, door_2_wall_id]
                ):
                    if np.sum(index) < 1:
                        continue
                    for (wx1, wy1), (wx2, wy2) in zip(
                        np.round(lpts)[index].astype(np.int32).tolist(),
                        np.round(rpts)[index].astype(np.int32).tolist(),
                    ):
                        new_wd.append([(wx1, wy1), (wx2, wy2)])
                        id_buffer.append(ii)
        lines[1] = new_window
        lines[2] = new_door
        return lines, window_2_wall_id, door_2_wall_id

    @staticmethod
    def get_axisloc(line_pt1, line_pt2, xaxis=None, yaxis=None):
        x1, y1 = line_pt1
        x2, y2 = line_pt2
        pts_for_inputx = []
        pts_for_inputy = []
        if abs(x1 - x2) == 0:
            if xaxis is not None:
                raise ValueError('Line is parallel to Y-axis, all Y-values is right')
            if yaxis is not None:
                pts_for_inputy = [(x1, y) for y in yaxis]
        else:
            if abs(y1 - y2) == 0:
                if xaxis is not None:
                    pts_for_inputx = [(x, y1) for x in xaxis]
                if yaxis is not None:
                    raise ValueError('Line is parallel to X-axis, all X-values is right')
            else:
                if xaxis is not None:
                    xaxis = np.float32(xaxis)
                    yrst = (y1 + (xaxis - x1) * (y2 - y1) / (x2 - x1))
                    pts_for_inputx = [(x, y) for x, y in zip(xaxis.tolist(), yrst.tolist())]
                if yaxis is not None:
                    yaxis = np.float32(yaxis)
                    xrst = (x1 + (yaxis - y1) * (x2 - x1) / (y2 - y1))
                    pts_for_inputy = [(x, y) for x, y in zip(xrst.tolist(), yaxis.tolist())]
        return pts_for_inputx, pts_for_inputy


    def get_lines_of_lnmask(self, lnmask, mpts, raw_scale=1.0, draw_line=False, is_wall=False):
        small_lnmask, small_pts, fsize = self.down_sample(
            lnmask, mpts, size_threshold=800)
        # infer lines at small lnmask
        skeleton = morphology.skeletonize(
            small_lnmask, method='lee').astype(np.uint8)  # 骨架提取
        edges = self.split_skeleton(skeleton)
        if is_wall:
            edges = self.remove_isolated(edges, threshold=5)
        all_edges = []
        for edge in edges:
            all_edges.extend(
                self.split_edge(edge))
        # fitting and mapping size to lnmask
        if is_wall:
            lines, raw_lines = self.fit_edges(
                all_edges, small_pts, fx=fsize, fy=fsize, raw_scale=raw_scale,
                dis_threshold=5
            )
        else:
            lines, raw_lines = self.fit_opening_edges(
                all_edges, small_lnmask, small_pts, fx=fsize, fy=fsize, raw_scale=raw_scale,
                dis_threshold=5, score_thresh=0.55
            )
        vis_mask = None
        if draw_line:
            # vis_mask = self.show_edges(all_edges, skeleton)
            vis_mask = self.show_lines(lines, skeleton)
        return raw_lines, vis_mask


    def remove_isolated(self, edges, threshold):
        # if neither ending point of an edge has at least 2 neighbors(including itself),
        # this edge is judged as the isolated edge to be removed.
        ending_pts = []
        for edge in edges:
            (x1, y1), (x2, y2) = edge[0], edge[-1]
            ending_pts.extend([(x1, y1), (x2, y2)])
        matrix = (self.get_dis_matrix(ending_pts) <= threshold).astype(np.int32)
        ignore_same_edge_ending = [(2 * i, 2 * i + 1) for i in range(len(edges))] + \
                                  [(2 * i + 1, 2 * i) for i in range(len(edges))]
        ig_ids = np.int32(ignore_same_edge_ending)
        matrix[ig_ids[:, 1], ig_ids[:, 0]] = 0
        neighbors = np.sum(matrix, axis=1).reshape((-1, 2))
        retained_ids = np.sum((neighbors >= 2), axis=1) >= 1
        new_edges = [e for id, e in zip(retained_ids.tolist(), edges) if id]
        return new_edges


    @staticmethod
    def project_points(points, line_pt1, lint_pt2):
        """
        # for line: ax+by+c = 0 and point p0=(x0, y0),
        # the prejection of p0 is (p_x, p_y).
        #  m=(-b*X0 + a*Y0), n=-a**2-b**2
        #  p_x = (b*m + a*c)/n,  p_y = (b*c - a*m)/n

        Args:
            points, N*2
            line_pt1, (x, y)
            line_pt2, (x, y)
        Return:
            projects, N*2.
        """
        x1, y1 = line_pt1
        x2, y2 = lint_pt2
        a = y2 - y1
        b = x1 - x2
        c = y1 * x2 - y2 * x1
        n = -a ** 2 - b ** 2
        points = np.float32(points)
        x0, y0 = points[:, 0], points[:, 1]
        m = (-b * x0 + a * y0)
        p_x = ((b * m + a * c) * 1.0 / n).tolist()
        p_y = ((b * c - a * m) * 1.0 / n).tolist()
        projects = [(x, y) for x, y in zip(p_x, p_y)]
        return projects

    @staticmethod
    def fit_edge(edge):
        """vh type 水平墙: 0, 竖墙: 1, others:-1"""
        edge_float = np.float32(edge)
        cx, cy = np.mean(edge_float, axis=0).tolist()
        (x1, y1), (x2, y2) = edge[0], edge[-1]
        raw_endings = [(x1, y1), (x2, y2)]
        if len(edge) > 13:
            (x1, y1) = np.mean(edge[:4], axis=0).tolist()
            (x2, y2) = np.mean(edge[-4:], axis=0).tolist()
            #(x1, y1), (x2, y2) = edge[0], edge[-1]

        if abs(x2 - x1) * 10 <= abs(y2 - y1):
            projects = [(cx, y1), (cx, y2)]
            vh_type = V_TYPE
        elif abs(x2 - x1) * 0.1 > abs(y2 - y1):
            projects = [(x1, cy), (x2, cy)]
            vh_type = H_TYPE
        else:
            projects = SkeletonParser.project_points(
                [edge[0], edge[-1]],
                (x1, y1),
                (x2, y2)
            )
            vh_type = OTHERS
        return projects, raw_endings, vh_type

    def get_dis_matrix(self, pts):
        """Get distances between each two points."""
        pts_len = len(pts)
        if pts_len < 1:
            return None
        if isinstance(pts, list):
            pts = np.float32(pts)
        deta_xy = np.repeat(pts[None, ...], pts_len, axis=0) - \
                  np.repeat(pts[:, None, :], pts_len, axis=1)
        deta_xy = deta_xy.astype(np.float32)
        deta_xy[..., 0] += 1e-8
        z = deta_xy[..., 0] + deta_xy[..., 1] * 1j
        distances = np.abs(z)
        return distances

    def fit_pts_by_distance(self, pts, distance_matrix, threshold=5, ptmask_pts=None):
        match_index = (distance_matrix < threshold)
        pts_array = np.float32(pts)
        index_set = set(range(len(pts)))
        loc_mapping = {}
        while len(index_set) > 0:
            id = index_set.pop()
            mid = match_index[id]
            cluster_pts = pts_array[mid]
            id_set = np.where(mid)[0]
            id_set = set(id_set.tolist())
            index_set = index_set - id_set
            x, y = np.mean(cluster_pts, axis=0).tolist()
            for i in id_set:
                loc_mapping[i] = (x, y)
        assert len(loc_mapping) == len(pts)

        if ptmask_pts is not None:
            self.map_lmaskpts_2_ptmask(loc_mapping, ptmask_pts)
        self.align_lines(match_index, loc_mapping)
        return loc_mapping

    def align_lines(self, match_index, loc_mapping):
        """ Update line's common points to align line to axis.

        Args:
            match_index, N*N matching bool values among all key points.
            loc_mapping, N items, mapping key id to (x, y), the value set
                       is all lines' common points.
        """
        mapping_xloc = {}
        mapping_yloc = {}

        for catch_type in [V_TYPE, H_TYPE]:
            id_sets = set(range(match_index.shape[0]))
            while len(id_sets) > 0:
                i = id_sets.pop()
                cur_vhpts = {loc_mapping[i]}
                find_vh = {i}
                while len(find_vh) > 0:
                    id_set = np.where(
                        match_index[find_vh.pop()]
                    )[0]
                    id_set = set(id_set.tolist())
                    id_sets = id_sets - id_set # remove current locations

                    while len(id_set) > 0:
                        common_pt_j = id_set.pop()
                        edgej_stpt = loc_mapping[common_pt_j]
                        if common_pt_j % 2 == 0:
                            endpt_id = common_pt_j + 1
                        else:
                            endpt_id = common_pt_j - 1
                        edgej_endpt = loc_mapping[endpt_id]
                        id_sets -= {endpt_id}  # remove edge end point
                        if edgej_endpt in cur_vhpts:
                            continue
                        _, _, vh_type = self.fit_edge([edgej_stpt, edgej_endpt])
                        if vh_type == catch_type:
                            cur_vhpts.add(edgej_endpt)
                            find_vh.add(endpt_id)
                if catch_type == V_TYPE:
                    x_locs = [x for (x, y) in cur_vhpts]
                    x_locs_mean = sum(x_locs) * 1.0 / len(x_locs)
                    for pt in cur_vhpts:
                        mapping_xloc[pt] = x_locs_mean
                elif catch_type == H_TYPE:
                    y_locs = [y for (x, y) in cur_vhpts]
                    y_locs_mean = sum(y_locs) * 1.0 / len(y_locs)
                    for pt in cur_vhpts:
                        mapping_yloc[pt] = y_locs_mean
                else:
                    # TODO
                    raise NotImplementedError('')
        # update mapping dict
        for k, pt in loc_mapping.items():
            x = mapping_xloc.get(pt, pt[0])
            y = mapping_yloc.get(pt, pt[1])
            loc_mapping[k] = (x, y)
        return


    def fit_vh_pts_by_distance(self, pts, distance_matrix, threshold=5):
        match_index = (distance_matrix < threshold)
        pts_array = np.float32(pts)
        index_set = set(range(len(pts)))
        loc_mapping = {}
        while len(index_set) > 0:
            id = index_set.pop()
            mid = match_index[id]
            cluster_pts = pts_array[mid]
            id_set = np.where(mid)[0]
            id_set = set(id_set.tolist())
            index_set = index_set - id_set
            x, y = np.mean(cluster_pts, axis=0).tolist()
            for i in id_set:
                loc_mapping[i] = (x, y)
        assert len(loc_mapping) == len(pts)

        return loc_mapping


    def mapping_lines_from_locs(self, loc_mapping, fx=1.0, fy=1.0, raw_scale=1.0):
        keys = loc_mapping.keys()
        lines, raw_lines = [], []
        for i in range(len(keys)//2):
            for backet, scale in zip(
                [lines, raw_lines], [1.0, raw_scale]
            ):
                pt1 = np.round(
                    np.float32(loc_mapping[2 * i]) * np.float32([scale * fx, scale * fy])
                ).astype(np.int32).tolist()
                pt2 = np.round(
                    np.float32(loc_mapping[2 * i + 1]) * np.float32([scale * fx, scale * fy])
                ).astype(np.int32).tolist()
                backet.append(
                    [tuple(pt1), tuple(pt2)]
                )
        return lines, raw_lines


    def fit_edges(self, edges, ptmask_pts, fx=1.0, fy=1.0, raw_scale=1.0, dis_threshold=5):
        """Infer lines from edges, all intersected lines have commom ending points."""
        lines, raw_lines = [], []
        if len(edges) < 1:
            return lines, raw_lines

        proj_pts = []
        for edge in edges:
            # proj, _, _ = self.fit_edge(edge)
            # proj_pts.extend(proj)
            (x1, y1), (x2, y2) = edge[0], edge[-1]
            proj_pts.extend([(x1, y1), (x2, y2)])
        distance_matrix = self.get_dis_matrix(proj_pts)
        loc_mapping = self.fit_pts_by_distance(
            proj_pts, distance_matrix, dis_threshold, ptmask_pts)
        lines, raw_lines = self.mapping_lines_from_locs(loc_mapping, fx=fx, fy=fy, raw_scale=raw_scale)
        return lines, raw_lines

    def fit_opening_edges(self, edges, lnmask, ptmask_pts, fx=1.0, fy=1.0, raw_scale=1.0, dis_threshold=5, score_thresh=0.6):
        """Infer lines from door/window edges, all intersected lines have commom ending points."""
        lines, raw_lines = [], []
        if len(edges) < 1 or len(ptmask_pts) < 1:
            return lines, raw_lines

        ptmask_pts_array = np.float32(ptmask_pts)
        for edge in edges:
            (x1, y1), (x2, y2) = edge[0], edge[-1]
            distance = self.distance(ptmask_pts, (x1, y1), (x2, y2))
            coase_match = distance < dis_threshold
            coase_ids = np.where(coase_match)[0]
            if np.sum(coase_match) < 2:
                continue
            match_ids = []
            for pt in [(x1, y1), (x2, y2)]:
                nearest_id = np.argmin(
                    np.sum(np.abs(ptmask_pts_array[coase_match] - np.float32(pt)), axis=1)
                )
                match_ids.append(
                    coase_ids[nearest_id]
                )
            if match_ids[0] == match_ids[1]:
                continue
            (nx1, ny1), (nx2, ny2) = ptmask_pts[match_ids[0]], ptmask_pts[match_ids[1]]
            arg_loc = tuple(np.round([nx1, ny1, nx2, ny2]).astype(np.int32).tolist())
            cc, rr = drawer.line(*arg_loc) #nx1, ny1, nx2, ny2)
            score = np.mean(lnmask[rr, cc])
            if score < score_thresh:
                continue
            for backet, scl in zip(
                [lines, raw_lines],
                [1.0, raw_scale]
            ):
                backet.append(
                    [tuple(np.round((nx1 * fx * raw_scale, ny1 * fy * raw_scale)).astype(np.int32).tolist()),
                     tuple(np.round((nx2 * fx * raw_scale, ny2 * fy * raw_scale)).astype(np.int32).tolist()),
                     # score
                     ]
                )
        return lines, raw_lines

    def map_lmaskpts_2_ptmask(self, loc_map, ptmask_pts):
        key_pts = set(list(loc_map.values()))
        key_pts = list(key_pts)
        if len(ptmask_pts) < 1:
            return
        deta_xy = np.repeat(np.float32(key_pts)[None, ...], len(ptmask_pts), axis=0) - \
                  np.repeat(np.float32(ptmask_pts)[:, None, :], len(key_pts), axis=1)
        z = deta_xy[..., 0] + deta_xy[..., 1] * 1j
        distances = np.abs(z)
        min_index = np.argmin(distances, axis=0).tolist()
        min_dises = np.min(distances, axis=0).tolist()
        trans_dict = {}
        for i in range(len(key_pts)):
            if min_dises[i] > 5:
                continue
            j = min_index[i]
            trans_dict[key_pts[i]] = ptmask_pts[j]
        for k, v in loc_map.items():
            loc_map[k] = trans_dict.get(v, v)
        return

    @staticmethod
    def on_skeleton(nx, ny, h, w, sk_mask):
        out_side = min(nx, ny) < 0 or nx >= w or ny >= h
        if out_side:
            return False
        return sk_mask[ny, nx] != 0

    def get_skeleton_keypts(self, skeleton):
        h, w = skeleton.shape[:2]
        yy, xx = np.where(skeleton > 0)
        points = []
        type2pts = defaultdict(set)
        pts2type = dict()
        for x, y in zip(xx.tolist(), yy.tolist()):
            vh4_pts = [
                (x + dx, y + dy) for dx, dy in VH4_NBRS
                if self.on_skeleton(x + dx, y + dy, h, w, skeleton)
            ]
            x4_pts = [
                (x + dx, y + dy) for dx, dy in X4_NBRS
                if self.on_skeleton(x + dx, y + dy, h, w, skeleton)
            ]
            type2pts[(len(vh4_pts), len(x4_pts))].add((x, y))
            points.append((x, y))
            pts2type[(x, y)] = (len(vh4_pts), len(x4_pts))
        return points, type2pts, pts2type

    def split_skeleton(self, skeleton, points=None, type2pts=None, pts2type=None):
        edges = []
        skt_cp = np.copy(skeleton)
        h, w = skeleton.shape[:2]
        if points is None:
            points, type2pts, pts2type = self.get_skeleton_keypts(skeleton)
        if len(points) < 1:
            return edges

        pts_set = set(points)
        while len(pts_set) > 0:
            # select max value of crossing
            cross_types = list(type2pts.keys())
            max_type = max(cross_types)

            # pop a starting point
            max_tp_pts = type2pts[max_type]
            max_tp_pts = max_tp_pts.intersection(pts_set)
            if len(max_tp_pts) == 0:
                type2pts.pop(max_type)
                continue
            (x, y) = max_tp_pts.pop()
            type2pts[max_type] = max_tp_pts
            if len(max_tp_pts) == 0:
                type2pts.pop(max_type)

            edges_cur = self.walk_along_edge(
                (x, y), pts_set, pts2type, skt_cp, h, w)
            edges.extend(edges_cur)
        return edges

    @staticmethod
    def edge_distances(edge_pts):
        return SkeletonParser.distance(edge_pts, edge_pts[0], edge_pts[-1])

    @staticmethod
    def distance(points, line_pt1, line_pt2):
        """
        Args:
            points, N*2
            line_pt1, (x, y)
            line_pt2, (x, y)
        Return:
            distance, 1D-array with N values.
        """
        points = np.float32(points)
        line_point1 = np.float32(line_pt1)
        line_point2 = np.float32(line_pt2)
        vec1 = line_point1 - points
        vec2 = line_point2 - points
        distance = np.abs(np.cross(vec1, vec2)) / (np.linalg.norm(line_point1 - line_point2) + 1e-6)
        distance = np.round(distance, 2)
        return distance


    def split_edge(self, edge_pts):
        if len(edge_pts) < 5:
            return [] #return [edge_pts]
        edges = []
        distance = self.edge_distances(edge_pts)
        max_dis = np.max(distance)
        if max_dis >= self.edge_max_dis:
            index = np.argmax(distance)
            edges.extend(
                self.split_edge(
                    edge_pts[:index+1])
            )
            edges.extend(
                self.split_edge(edge_pts[index:]))
        else:
            edges.append(edge_pts)
        return edges

    def split_edge_by_pts_num(self, edge_pts, pts_num=5, edge_max_dis=2):
        if len(edge_pts) < 2 * pts_num:
            edge_pts = F.interpolate(
                torch.as_tensor(edge_pts, dtype=torch.float32)[None, None, ...],
                size=(pts_num, 2),
                mode='bilinear', align_corners=True
            ).numpy()[0, 0]
        edges = [edge_pts]
        while len(edges) + 1 < pts_num:
            dis_list = [self.edge_distances(per_edge)
                        for per_edge in edges]
            max_ids = [np.argmax(dis) for dis in dis_list]
            maxdis_length = [
                (dis[id], len(_edge)) if (dis[id] > edge_max_dis) else (0, len(_edge))
                for id, dis, _edge in zip(max_ids, dis_list, edges)]
            edge_id = maxdis_length.index(max(maxdis_length))
            _edge = edges[edge_id]
            if maxdis_length[edge_id][0] <= edge_max_dis:
                _index = len(_edge) // 2
            else:
                _index = max_ids[edge_id]
            edges = edges[: edge_id] + [_edge[: _index + 1], _edge[_index:]] + edges[edge_id + 1:]
        return edges

    def split_edge_by_maxdis(self, edge_pts, edge_max_dis=3, ignore_small=5):
        edges = [edge_pts]
        _max_dis = edge_max_dis
        while _max_dis >= edge_max_dis:
            dis_list = [self.edge_distances(per_edge)
                        for per_edge in edges]
            max_ids = [np.argmax(dis) for dis in dis_list]
            maxdis_length = [(dis[id], len(_edge)) for id, dis, _edge in zip(max_ids, dis_list, edges)]
            edge_id = maxdis_length.index(max(maxdis_length))
            _edge = edges[edge_id]
            _index = max_ids[edge_id]
            _max_dis = maxdis_length[edge_id][0]
            if _max_dis >= edge_max_dis:
                edges = edges[: edge_id] + [_edge[: _index], _edge[_index:]] + edges[edge_id + 1:]
        return edges

    def split_edge_to_samesegs(self, edge_pts, per_segment_len=7):
        assert per_segment_len >= 2, f"per_segment_len: {per_segment_len}"
        _edges = self.split_edge_by_maxdis(edge_pts, edge_max_dis=3, ignore_small=3)
        edges = []
        for edge in _edges:
            _len = len(edge)
            segs = int(round(_len * 1.0 / per_segment_len))
            segs = max(segs, 1)
            edge_pts = F.interpolate(
                torch.as_tensor(edge, dtype=torch.float32)[None, None, ...],
                size=(segs * per_segment_len + 1, 2),
                mode='bilinear', align_corners=True
            ).numpy().tolist()[0][0]
            edges.extend([
                edge_pts[i * per_segment_len: (i + 1) * per_segment_len + 1]
                for i in range(segs)
            ])
        return edges


    def get_farthest_pt(self, edge_pts, dis_thld=2):
        distance = self.edge_distances(edge_pts)
        max_dis_id = np.argmax(distance)
        if distance[max_dis_id] < dis_thld:
            max_dis_id = len(edge_pts) // 2
        return edge_pts[max_dis_id]

    def edge_to_curves_by_distance(self, edge_pts, dis_thld=2):
        if len(edge_pts) < 5:
            return [edge_pts]
        edges = []
        distance = self.edge_distances(edge_pts)
        max_dis_id = np.argmax(distance)

        if distance[max_dis_id] < dis_thld:
            # max_dis_id = len(edge_pts) // 2
            edges.append(edge_pts)
        else:
            #if max_dis >= self.edge_max_dis:
            edges.extend(
                self.split_edge(
                    edge_pts[:max_dis_id+1])
            )
            edges.extend(
                self.split_edge(edge_pts[max_dis_id:]))
        return edges

    def edge_to_curves_by_curvature(self, edge_pts):
        def curvature_by_chord_farthest_pt(curve_pts):
            curve_pts = np.float32(curve_pts)
            distance = self.edge_distances(curve_pts)
            max_dis_id = np.argmax(distance)
            curvature = 0
            if distance[max_dis_id] != 0:
                mid_id = len(distance) // 2
                mid_dis = 0.5 * (distance[max_dis_id] + distance[mid_id])
                chord_len = np.linalg.norm(curve_pts[0] - curve_pts[-1])
                #r = (mid_dis ** 2 + 0.25 * chord_len ** 2) / (2 * mid_dis)
                curvature = (8 * mid_dis) / (4 * mid_dis ** 2 + chord_len ** 2)
            return curvature

        # edge_pts = np.float32(edge_pts)
        arc_length = 11
        half_arc = arc_length // 2
        curvature = 0
        curvs = np.zeros(shape=(len(edge_pts),), dtype=np.float32)
        for i in range(half_arc, len(edge_pts) - half_arc):
            curvature = curvature_by_chord_farthest_pt(edge_pts[i - half_arc: i - half_arc + arc_length + 1])
            if i == half_arc:
                curvs[: i + 1] += curvature
            else:
                curvs[i] = curvature
        curvs[-half_arc:] += curvature
        win = signal.hanning(23)
        curvs_n = signal.convolve(curvs, win, mode='same') / sum(win)
        peak_ids = signal.find_peaks(curvs_n)[0].tolist() #+ signal.find_peaks(-curvs_n)[0].tolist()
        peak_ids.sort()
        st = [0] + peak_ids
        ed = peak_ids + [len(edge_pts)]
        curves = [edge_pts[i:j] for i, j in zip(st, ed)]
        return curves

    def edge_to_small_curves_by_curvature(self, edge_pts, length=-1):
        curves = self.edge_to_curves_by_curvature(edge_pts)
        if length < 1:
            return curves
        else:
            n_curves = []
            for c in curves:
                split_num = (len(c) // length) + (1 if (len(c) % length) > (0.8 * length) else 0)
                if split_num <= 1:
                    n_curves.append(c)
                else:
                    for i in range(split_num):
                        n_curves.append(c[i * length: (i + 1) * length + 1])
            return n_curves

    def walk_along_edge(self, start_pt, pts_set, pts2type, skeleton_cp, h, w):
        (x, y) = start_pt
        is_mid_pt = (sum(pts2type[(x, y)]) == 2)
        skeleton_cp[y, x] = 0
        pts_set.remove((x, y))

        # set the start point of edges for given crossing point
        edges_cur = []
        for dx, dy in EIGHT_NBRS:
            if not self.on_skeleton(x + dx, y + dy, h, w, skeleton_cp):
                continue
            edges_cur.append(
                [(x, y), (x + dx, y + dy)]  # add an edge
            )
            skeleton_cp[y + dy, x + dx] = 0
            pts_set.remove((x + dx, y + dy))

        # walk along each edge at current start point
        for edge in edges_cur:
            (x, y) = edge[-1]
            # walk along current edge
            while True:
                next_nb_pts = [
                    (x + dx, y + dy) for dx, dy in EIGHT_NBRS
                    if self.on_skeleton(x + dx, y + dy, h, w, skeleton_cp)
                ]
                if len(next_nb_pts) == 1:
                    x, y = next_nb_pts[0]
                    nb_pt = (x, y)
                    edge.append(nb_pt)
                    skeleton_cp[y, x] = 0
                    pts_set.remove((x, y))
                else:
                    # walk to the edge end while meeting the next crossing
                    break
        if is_mid_pt:
            assert len(edges_cur) <= 2
            if len(edges_cur) == 2:
                combine_edge = edges_cur[1][::-1] + edges_cur[0][1:]
                edges_cur = [combine_edge]
        return edges_cur


    def down_sample(self, lnmask, mpts, size_threshold=400):
        h, w = lnmask.shape[:2]
        while max(lnmask.shape) >= size_threshold:
            lnmask = (cv2.pyrDown(lnmask * 255) > 0).astype(np.uint8)
        fsize = max(h, w) * 1.0 / max(lnmask.shape)
        small_pts = np.float32(mpts) / fsize
        small_pts = [tuple(pt) for pt in small_pts.tolist()]
        return lnmask, small_pts, fsize

    def show_edges(self, edges, skeleton):
        h, w = skeleton.shape[:2]
        binary = np.zeros((h, w, 3), dtype=np.uint8)
        for per_e in edges:
            color = (
                np.random.randint(30, 255),
                np.random.randint(30, 255),
                np.random.randint(30, 255)
            )
            try:
                pts = np.int32(per_e)
            except:
                print(per_e[0])
                exit()
            binary[pts[:, 1], pts[:, 0]] = color
        return binary

    def show_lines(self, lines, skeleton, color=(255, 255, 255)):
        h, w = skeleton.shape[:2]
        binary = np.zeros((h, w, 3), dtype=np.uint8)
        for pt1, pt2 in lines:
            if color is None:
                color = (
                    np.random.randint(30, 255),
                    np.random.randint(30, 255),
                    np.random.randint(30, 255)
                )
            cv2.line(binary, pt1, pt2, color, thickness=5)
        return binary

class MaskParser(BaseParser):

    def __init__(self, angle_num=18, r=13, R=31):
        self.pol_masks = self.get_pol_masks(angle_num, r, R)
        # raise Warning('This class (MaskParser) is NOT for any downstream usage, '
        #               'since the author develop more efficient SkeletonParser, '
        #               'SkeletonParser is recommanded if you need a parser.')

    def get_pol_masks(self, angle_num=18, r=13, R=31):
        loc_len = 2 * R + 1
        cx, cy = R, R
        rs = R * np.ones(shape=(angle_num,), dtype=np.float32)
        angles = np.arange(start=0, stop=360, step=360 // angle_num)
        x, y = pol2cart(rs, angles)
        locus = np.zeros((angle_num, 2), dtype=np.float32)
        locus[:, 0] += x
        locus[:, 1] += y
        loc_array = locus.astype(np.int32)
        local_masks = np.zeros(shape=(angle_num, loc_len, loc_len), dtype=np.uint8)
        for mi, (dx, dy) in enumerate(loc_array.tolist()):
            x, y = cx + dx, cy + dy
            rr, cc = drawer.line(cx, cy, x, y)
            local_masks[mi][cc, rr] = 1
        pol_masks = local_masks[:, cy - r:cy + r + 1, cx - r:cx + r + 1]
        return pol_masks


    def get_lines_of_lnmask(self, lnmask, mpts, draw_line=False):
        """
        Args:
            lnmask, H*W, 0-1 np.array
            mpts, [(x, y), ... ]
            mpts_rawpts, {(x, y):(raw_x, raw_y), ... }
            pol_masks, angle_num * r * r, 0-1 np.array
        """
        pts_len = len(mpts)
        if pts_len < 1:
            return None, None
        pol_masks = self.pol_masks
        angle_num = pol_masks.shape[0]

        # Get directions for each point
        max_dirs, dirs_index = self.get_pts_directions(pol_masks, lnmask, mpts)

        # Get angle matrix
        angle_matrix = self.get_angle_matrix(mpts, angle_num)

        # Get candidate lines
        candidates = self._get_candidates(mpts, angle_matrix, dirs_index)

        # Get lines of short list
        lines = self._get_lines(mpts, candidates, lnmask)
        vis_mask = None
        if draw_line:
            h, w = lnmask.shape[:2]
            vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for pt1, pt2 in lines:
                cv2.line(vis_mask, pt1, pt2, (255, 255, 255), thickness=3)
        return lines, vis_mask


    def _get_candidates(self, pts, angle_matrix, dirs_index):
        """Acoording to directions, infer the long list of possible lines.

        Args:
            pts, [(x, y), ... ] with shape N*2
            angle_matrix, np.array with shape N*N, angle_matrix[i,j] is a quantatited angle
                       by angle numbers in  {pol_masks}, showing the angle between point i and
                       point j.
            dirs_index, 0-1 array, N*{angle_num}, 1 means the strictly predicted direction.

        Return:
            candidates, 0-1 np.array with shape N*N, value at [i,j] is a bool(int) to show
                  whether point i and point j can be a line.
        """
        pts_len = len(pts)
        if pts_len < 1:
            return None
        angle_num = dirs_index.shape[1]
        # use [1, angle_num+1) to avoid mixture between index-0 and and value-0
        dirs_index = dirs_index * np.arange(1, angle_num+1)
        candidates = np.zeros_like(angle_matrix)
        for i in range(pts_len):
            candidates[i] += angle_matrix[i]
            candidates[i, 0: i+1] = 0
            directions = set(dirs_index[i].tolist())
            directions.remove(0)
            per_cddt = np.zeros_like(candidates[i, i+1:])
            for dir_index in directions:
                per_cddt += (candidates[i, i+1:] == (dir_index - 1)).astype(np.int32)
            candidates[i, i+1:] = per_cddt
        return candidates


    def _get_lines(self, pts, candidates, lnmask, mpts_rawpts=None, threshold=0.85):
        """Given the long list of possible lines, the line-mask, and the locations of all points,
        infer the short list of line segments by pixel iou.

        Args:
            pts, [(x, y), ... ] with shape N*2
            candidates, 0-1 np.array with shape N*N, candidates[i,j] is a bool(int) value,
                       showing whether there may be a line between point i and point j.
            lnmask, H*W, 0-1 np.array of skeleton detections.
            mpts_rawpts, a mapping dict of points locations.

        Return:
            json_dict,
                {
                "Points":[{'x':x, 'y':y, 'id':id, 'score':score}, ...],
                'Lines':[{"start_point": 215, "end_point": 194, "id": 716}],
                }
            raw_data, [pred_pts, pred_lines],
               pred_pts is [(x, y), ... ] with shape (<=N)*2
               pred_lines is [[(x1, y1), (x2, y2)], ... ].
        """
        # TODO add code here
        pts_len = len(pts)
        if pts_len < 1:
            return None
        if np.sum(candidates) == 0:
            return None
        yy, xx = np.where(candidates > 0)
        ious = []
        lines = []
        for ii, jj in zip(yy.tolist(), xx.tolist()):
            new_mask = np.zeros_like(lnmask)
            x1, y1 = pts[ii]
            x2, y2 = pts[jj]
            cc, rr = drawer.line(x1, y1, x2, y2)
            new_mask[rr, cc] = 1
            intersects = np.sum(new_mask * lnmask).item()
            line_len = len(cc)
            pixel_iou = intersects * 1.0 / line_len
            ious.append(round(pixel_iou, 4))
            # if pixel_iou < 0.7 and pixel_iou >= 0.5:
            if pixel_iou >= threshold:
                lines.append((ii, jj))
        return lines


    def get_pts_directions(self, pol_masks, lnmask, pts):
        """Given the pol mask and line mask, points, infer the potential
        directions of each point.

        Args:
            pts, [(x, y), ... ]
            pol_masks, angle_num * r * r, 0-1 np.array
            lnmask, H*W, 0-1 np.array of skeleton detections.

        Return:
            max_direction_preds, int-array, N*{angle_num}, non-zero value
                                 means the potential direction score.
            dirs_index, 0-1 array, N*{angle_num}, 1 means the strictly predicted direction.
        """
        pts_len = len(pts)

        # padding mask to avoid negative positions
        r = pol_masks.shape[1] // 2
        angle_num = pol_masks.shape[0]
        h, w = lnmask.shape[:2]
        padding_lnmask = np.zeros(shape=(2 * r + h, 2 * r + w), dtype=np.uint8)
        padding_lnmask[r : r + h, r : r + w] += lnmask
        # adjust locations of points for padding_lnmask
        padding_pts = np.int32(pts) + np.int32([r, r])

        local_masks = np.repeat(pol_masks[None, ...], pts_len, axis=0)
        local_preds = np.concatenate([
            np.repeat(
                padding_lnmask[y - r: y + r + 1, x - r: x + r + 1][None, ...],
                angle_num,
                axis=0
            )[None, ...]
            for x, y in padding_pts.tolist()
        ], axis=0)
        anchor_preds = np.sum(
            np.sum(local_preds * local_masks, axis=3), axis=2).astype(np.int32)
        pool = torch.nn.MaxPool2d((1, 3), 1, padding=0)
        anchor_preds_padding = np.zeros(shape=(1, 1, pts_len, angle_num+2), dtype=np.int32)
        anchor_preds_padding[0, 0, :, 1: angle_num+1] += anchor_preds
        anchor_preds_padding[0, 0, :, 0:1] += anchor_preds[:, angle_num-2:angle_num-1]
        anchor_preds_padding[0, 0, :, angle_num+1:angle_num+2] += anchor_preds[:, 0:1]
        anchor_preds_padding = anchor_preds_padding.astype(np.float32)
        anchor_preds_padding = torch.from_numpy(anchor_preds_padding)
        maxm = pool(anchor_preds_padding)
        maxm = torch.eq(maxm, anchor_preds_padding[..., 1: angle_num+1]).int()
        max_direction_preds = anchor_preds * maxm[0, 0].numpy()
        threshold = np.max(max_direction_preds).item() // 2
        dirs_index = (max_direction_preds > max(r - 4, threshold)).astype(np.int32)
        return max_direction_preds, dirs_index


    def get_angle_matrix(self, pts, angle_num):
        """Get angles between each other of all points."""
        pts_len = len(pts)
        if pts_len < 1:
            return None
        if isinstance(pts, list):
            pts = np.int32(pts)
        deta_xy = np.repeat(pts[None, ...], pts_len, axis=0) - \
                  np.repeat(pts[:, None, :], pts_len, axis=1)
        deta_xy = deta_xy.astype(np.float32)
        deta_xy[..., 0] += 1e-8
        z = deta_xy[..., 0] + deta_xy[..., 1] * 1j
        per_angle = (360 // angle_num)
        angle_matrix = np.round(np.angle(z, True) / per_angle).astype(np.int32)
        angle_matrix += (angle_matrix < 0).astype(np.int32) * angle_num
        return angle_matrix


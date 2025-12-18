# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import skimage.draw as drawer
from skimage import morphology
from .sk_parser import SkeletonParser


DIL_KERNER = np.zeros((3, 3), dtype=np.uint8)
DIL_KERNER[:, 1] += 1
DIL_KERNER[1, :] += 1


class SkPts:

    parser = SkeletonParser()

    @staticmethod
    def get_pt_graph(skeleton):
        """
        Args:
            skeleton, np.array of 0-1 values

        Return:
            [endpoints, junctions, corners], each item is a list like [(x, y), ... ]
                    endpoints and junctions are points that are easy to be found in skeleton,
                    corners are points at a non-straight branch to compose line segments in
                    fitting this branch.
            adj_matrix, 2D np.array of values [-1, 0, 1], representing the adjacent matrix
                        1 means positive path, -1 means negative paths, 0 means ignore.
            mimic_branches, each item is a line segment to fit a branch in skeleton,
                            note that each branch is fitted by multiple straight line
                            segments.
            edges, [positive paths, negative paths], each item is [line_segment, ...]
        """
        def is_negative_path(mask, line):
            pt1, pt2 = line
            xx, yy = drawer.line(pt1[0], pt1[1], pt2[0], pt2[1])
            negative_pixels = len(xx) - np.sum(mask[yy, xx])
            return negative_pixels > 0

        new_sklt = cv2.dilate(skeleton, DIL_KERNER, iterations=4)
        endpoints, junctions, corners, straight_edges = SkPts.get_key_points(skeleton)
        pts = endpoints + junctions + corners
        if len(pts) < 1:
            return None, None, None

        edge_pts = []
        for e in straight_edges:
            edge_pts.extend([e[0], e[-1]])

        deta_xy = np.repeat(np.float32(edge_pts)[None, ...], len(pts), axis=0) - \
                  np.repeat(np.float32(pts)[:, None, :], len(edge_pts), axis=1)
        z = deta_xy[..., 0] + deta_xy[..., 1] * 1j
        distances = np.abs(z) # len(pts) * len(edge_pts)
        min_index = np.argmin(distances, axis=0).tolist()
        min_dises = np.min(distances, axis=0).tolist()
        trans_dict = {} # from straight pt to endpoints + junctions + corners
        for i in range(len(edge_pts)):
            if min_dises[i] > 5:
                warnings.warn(f'Bad min distance: {min_dises[i]} in mapping')
            j = min_index[i]
            trans_dict[edge_pts[i]] = (pts[j], j)
        mimic_branches = []
        edges = [[], []] # [positive paths, negative paths]
        adj_matrix = np.zeros((len(pts), len(pts)), dtype=np.int32)
        for ei, e in enumerate(straight_edges):
            node_i = trans_dict[e[0]][1]
            node_j = trans_dict[e[-1]][1]
            if node_i != node_j:
                adj_matrix[node_i, node_j] = 1
                adj_matrix[node_j, node_i] = 1
                mimic_branches.append(
                    [pts[node_i], pts[node_j]]
                )
                edges[0].append((pts[node_i], pts[node_j]))
                # edges[0].append((pts[node_j], pts[node_i]))

        for i in range(0, adj_matrix.shape[0]):
           for j in range(i + 1, adj_matrix.shape[0]):
               if adj_matrix[i, j] == 0:
                   if is_negative_path(new_sklt, [pts[i], pts[j]]):
                       backe_id = 1
                       # adj_matrix[i, j] = -1
                       # adj_matrix[j, i] = -1
                   else:
                       backe_id = 0
                       adj_matrix[i, j] = 1
                       adj_matrix[j, i] = 1
                   edges[backe_id].append((pts[i], pts[j]))
                   # edges[backe_id].append((pts[j], pts[i]))
        return [endpoints, junctions, corners], adj_matrix, mimic_branches, edges

    @staticmethod
    def get_main_branches(mask, do_extract=True, dil_iters=2):
        if do_extract:
            branches, _ = SkPts.get_main_branches_newskeleton(
                mask, dil_iters)
        else:
            skeleton = (mask > 0).astype(np.uint8)
            branches = SkPts.parser.split_skeleton(skeleton)
        return branches

    @staticmethod
    def get_main_branches_newskeleton(mask, dil_iters=2):
        skeleton = (mask > 0).astype(np.uint8)
        skeleton = cv2.dilate(
            skeleton, kernel=DIL_KERNER, iterations=dil_iters)
        skeleton = morphology.skeletonize(
            skeleton, method='lee').astype(np.uint8)
        branches = SkPts.parser.split_skeleton(skeleton)
        return branches, skeleton

    @staticmethod
    def is_junction_of_branch_sides(branches, dist_thresh=3, edpt_type=0, junc_type=1):
        """ Label the side points type of each branch in {branches}
        Args:
            branches: a list, [branch, ...], each branch is a list of points
            dist_thresh: int, to judge if two points are close enough

        Return:
            side_pts_type, array of shape (len(branches), 2), label the type
                           of all side points, 0 means endpoint, 1 means junctions
        """
        assert edpt_type != junc_type
        len_branches = len(branches)
        if len_branches < 1:
            return np.zeros(shape=(len_branches, 2), dtype=np.int32)

        proj_pts = []
        for edge in branches:
            (x1, y1), (x2, y2) = edge[0], edge[-1]
            proj_pts.extend([(x1, y1), (x2, y2)])
        dist_mat = 999 * np.eye(len_branches * 2) + \
                   SkPts.parser.get_dis_matrix(proj_pts)
        side_pts_type = np.zeros(shape=(len_branches * 2,), dtype=np.int32)
        dist_mat = (dist_mat < dist_thresh).astype(np.int32)
        neighbors = np.sum(dist_mat, axis=0)
        is_endpoint = (neighbors < 1)   # endpoints have non-neighbours close enough
        is_junction = (neighbors >= 1)  # junctions have neighbours close enough
        side_pts_type[is_endpoint] = edpt_type
        side_pts_type[is_junction] = junc_type
        side_pts_type = side_pts_type.reshape((len_branches, 2)).tolist()
        return side_pts_type

    @staticmethod
    def visualise_pts(mask, do_extract=True, show=True, dil_iters=2):
        if do_extract:
            skeleton = morphology.skeletonize(
                mask, method='lee').astype(np.uint8)
        else:
            skeleton = mask
        yy, xx = np.where(skeleton > 0)
        skpts = np.concatenate([xx[:, None], yy[:, None]], axis=1)
        endpoints, junctions, corners, straight_edges = SkPts.get_key_points(skeleton)
        endpoints, junctions, corners = np.int32(endpoints), \
                                        np.int32(junctions), \
                                        np.int32(corners)
        h, w = skeleton.shape[:2]
        sk_cps = np.zeros((4, h, w), dtype=np.uint8)
        if len(endpoints) > 0:
            sk_cps[0][endpoints[:, 1], endpoints[:, 0]] = 1
        if len(junctions) > 0:
            sk_cps[1][junctions[:, 1], junctions[:, 0]] = 1
        if len(corners) > 0:
            sk_cps[2][corners[:, 1], corners[:, 0]] = 1
        sk_cps[3] += mask
        sk_cps[3] = cv2.dilate(sk_cps[3], kernel=DIL_KERNER, iterations=1)
        if dil_iters > 0:
            for i in range(3):
                sk_cps[i] = cv2.dilate(sk_cps[i], kernel=DIL_KERNER, iterations=dil_iters)

        if show:
            pylab.rcParams['figure.figsize'] = (12, 9)
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(sk_cps[3])
            plt.title('raw skeleton')
            plt.subplot(2, 3, 2)
            plt.imshow(sk_cps[0])
            plt.title('endpoints')
            plt.subplot(2, 3, 3)
            plt.imshow(sk_cps[1])
            plt.title('junctions')
            plt.subplot(2, 3, 4)
            plt.imshow(sk_cps[2])
            plt.title('corners')
            plt.subplot(2, 3, 5)
            plt.imshow((sk_cps[0] + sk_cps[1] + sk_cps[2]) > 0)
            plt.title('all pts')
            plt.show()
            plt.close()

        return sk_cps, endpoints, junctions, corners, skpts

    @staticmethod
    def visualise_graph(mask, do_extract=True, show=True, dil_iters=2, sample_num=0):
        if do_extract:
            skeleton = morphology.skeletonize(
                mask, method='lee').astype(np.uint8)
        else:
            skeleton = mask
        pts_list, adj_matrix, mimic_branches, [pos_paths, nega_paths] = SkPts.get_pt_graph(skeleton)
        endpoints, junctions, corners = pts_list
        endpoints, junctions, corners = np.int32(endpoints), \
                                        np.int32(junctions), \
                                        np.int32(corners)
        if sample_num > 0: # sample partial paths among all
            if sample_num < len(pos_paths):
                pos_paths = random.sample(pos_paths, sample_num)
            if sample_num < len(nega_paths):
                nega_paths = random.sample(nega_paths, sample_num)
        pos_paths, nega_paths = np.int32(pos_paths), np.int32(nega_paths)
        h, w = skeleton.shape[:2]
        sk_cps = np.zeros((7, h, w), dtype=np.uint8)
        if len(endpoints) > 0:
            sk_cps[0][endpoints[:, 1], endpoints[:, 0]] = 1
        if len(junctions) > 0:
            sk_cps[1][junctions[:, 1], junctions[:, 0]] = 1
        if len(corners) > 0:
            sk_cps[2][corners[:, 1], corners[:, 0]] = 1
        sk_cps[3] += mask
        if dil_iters > 0:
            for i in range(3):
                sk_cps[i] = cv2.dilate(sk_cps[i], kernel=DIL_KERNER, iterations=dil_iters)
        for m_branch in mimic_branches:
            pt1, pt2 = m_branch
            xx, yy = drawer.line(pt1[0], pt1[1], pt2[0], pt2[1])
            sk_cps[4, yy, xx] = 1
        if dil_iters > 0:
            sk_cps[4] = cv2.dilate(sk_cps[4], kernel=DIL_KERNER, iterations=max(dil_iters // 2, 1))
        sk_cps[5] = sk_cps[4]

        if show:
            pos_paths, nega_paths = pos_paths.tolist(), nega_paths.tolist()
            posi_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for posi_path in pos_paths:
                cv2.line(posi_mask, tuple(posi_path[0]), tuple(posi_path[1]),
                         (random.randint(120, 255), random.randint(120, 255), random.randint(120, 255)), thickness=21)
            nega_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for nega_path in nega_paths[:64]:
                cv2.line(nega_mask, tuple(nega_path[0]), tuple(nega_path[1]),
                         (random.randint(120, 255), random.randint(120, 255), random.randint(120, 255)), thickness=3)
            pylab.rcParams['figure.figsize'] = (12, 9)
            plt.figure()
            plt.subplot(2, 4, 1)
            plt.imshow(sk_cps[3])
            plt.title('raw skeleton')
            plt.subplot(2, 4, 2)
            plt.imshow(sk_cps[0])
            plt.title('endpoints')
            plt.subplot(2, 4, 3)
            plt.imshow(sk_cps[1])
            plt.title('junctions')
            plt.subplot(2, 4, 4)
            plt.imshow(np.transpose(sk_cps[3:6], [1, 2, 0]) * 255)
            plt.title('mimic branches')
            plt.subplot(2, 4, 5)
            plt.imshow(sk_cps[2])
            plt.title('corners')
            plt.subplot(2, 4, 6)
            plt.imshow((sk_cps[0] + sk_cps[1] + sk_cps[2]) > 0)
            plt.title('all pts')

            plt.subplot(2, 4, 7)
            plt.imshow(posi_mask)
            plt.title('posi paths')
            plt.subplot(2, 4, 8)
            plt.imshow(nega_mask)
            plt.title('nega paths')
            plt.show()
            plt.close()

        return sk_cps[:5], endpoints, junctions, corners, adj_matrix, pos_paths, nega_paths

    @staticmethod
    def visualise_mask(mask, do_extract=True, dil_iters=1):
        if do_extract:
            skeleton = morphology.skeletonize(
                mask, method='lee').astype(np.uint8)
        else:
            skeleton = mask
        skeleton = cv2.dilate(skeleton, kernel=DIL_KERNER, iterations=dil_iters)
        return skeleton

    @staticmethod
    def get_end_points(skeleton, do_extract=True, with_junctions=False):
        if do_extract:
            skeleton = morphology.skeletonize(skeleton, method='lee').astype(np.uint8)
        endpoints = []
        points, type2pts, pts2type = SkPts.parser.get_skeleton_keypts(skeleton)
        endpoints.extend(type2pts[(0, 1)])
        endpoints.extend(type2pts[(1, 0)])
        if with_junctions:
            junctions = SkPts.dedupe_junctions(type2pts)
            return endpoints, junctions
        else:
            return endpoints

    @staticmethod
    def get_key_points(skeleton):
        endpoints, junctions, corners = [], [], []

        points, type2pts, pts2type = SkPts.parser.get_skeleton_keypts(skeleton)
        endpoints.extend(type2pts[(0, 1)])
        endpoints.extend(type2pts[(1, 0)])
        junctions = SkPts.dedupe_junctions(type2pts)

        edges = SkPts.parser.split_skeleton(
            skeleton, points=points, type2pts=type2pts, pts2type=pts2type)

        straight_edges = []
        for edge in edges:
            new_edges = SkPts.parser.split_edge(edge)
            if len(new_edges) < 2:
                straight_edges.append(edge)
                continue
            else:
                straight_edges.extend(new_edges)
            raw_endpts = [edge[0], edge[-1]]
            new_endpts = []
            for e in new_edges:
                new_endpts.extend([e[0], e[-1]])
            new_endpts = set(new_endpts) - set(raw_endpts)
            corners.extend(list(new_endpts))
        if len(corners) > 1:
            corners = SkPts.dedupe_points(corners)
        return endpoints, junctions, corners, straight_edges

    @staticmethod
    def dedupe_points(pts):
        new_pts = []
        distances = SkPts.parser.get_dis_matrix(pts)
        dis_index = distances < 3
        max_nbrs = np.argsort(
            np.sum(dis_index, axis=1))[::-1]
        pts_ids_set = {i for i in range(len(pts))}
        new_ids_set = set()
        for cluster_index in max_nbrs.tolist():
            if cluster_index in new_ids_set:
                continue
            new_pts.append(pts[cluster_index])
            ids = np.where(dis_index[cluster_index])[0]
            cur_set = set(ids.tolist())
            new_ids_set = new_ids_set.union(cur_set)
            pts_ids_set -= cur_set
            if len(pts_ids_set) < 1:
                break
        return new_pts

    @staticmethod
    def dedupe_junctions(type2pts):
        candidates = []
        for k, pts in type2pts.items():
            # skip endpoints
            if k == (0, 1) or k ==(1, 0):
                continue
            # skip points like ---pt----
            if k == (1, 1) or k ==(2, 0) or k ==(0, 2):
                continue
            candidates.extend(pts)
        if len(candidates) > 1:
            junctions = SkPts.dedupe_points(candidates)
        else:
            junctions = candidates
        return junctions

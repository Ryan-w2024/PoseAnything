# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import os
import numpy as np
from skimage import morphology


def get_f1(precision, recall):
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

class SkeletonEvaluator(object):

    def __init__(self):
        self.predictions = []

    def update(self, predictions):
        """
        Args:
            predictions: [(gt_skeleton, pred_mask, im_path), ... ], each item is np.uint8 array
        """
        self.predictions.extend(predictions)

    def vis_only(self, visual_dir):
        assert visual_dir is not None and os.path.isdir(visual_dir)
        for gt_target, pred, im_path in self.predictions:
            pred_mask = morphology.skeletonize(
                pred, method='lee').astype(np.uint8)
            h, w = gt_target.shape
            assert (gt_target.shape == pred.shape)
            img = cv2.imread(im_path, 1)
            dks = 3
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dks, dks))
            pred_mask = cv2.dilate(pred_mask, element)
            pred_mask = (cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR) > 0)
            img = img * (1 - pred_mask)
            img[:, :, 2] = img[:, :, 2] + 255 * pred_mask[:, :, 0]
            img[:, :, 1] = img[:, :, 1] + 255 * pred_mask[:, :, 0]
            fileid = os.path.basename(im_path).split('.')[0]
            cv2.imwrite(os.path.join(visual_dir, f"{fileid}.png"), img)

    def summarize(self, score_threshold=0.5, offset_threshold=0.01, visual_dir=None):
        metrics = {'precision': [], 'recall': [], 'f1': []}
        for gt_target, pred, im_path in self.predictions:
            pred_mask = morphology.skeletonize(
                pred, method='lee').astype(np.uint8)
            h, w = gt_target.shape
            assert (gt_target.shape == pred.shape)
            if visual_dir is not None and os.path.isdir(visual_dir):
                # img = cv2.imread(im_path, 1)
                # vis_im = np.concatenate(
                #     [img,
                #      cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR),
                #      cv2.cvtColor(gt_target, cv2.COLOR_GRAY2BGR)],
                #     axis=1
                # )
                fileid = os.path.basename(im_path).split('.')[0]
                cv2.imwrite(os.path.join(visual_dir, f"{fileid}.png"), pred_mask)
            # plt.subplot(1, 2, 1)
            # plt.imshow(pred_mask)
            # plt.subplot(1, 2, 2)
            # plt.imshow(target)
            # plt.show()
            # plt.close()
            pred_yy, pred_xx = np.where(pred_mask)
            if len(pred_yy) < 1:
                precision, recall, f1 = 0, 0, 0
            else:
                pd_pts = np.stack([pred_xx, pred_yy], axis=1)
                gt_yy, gt_xx = np.where(gt_target > 0)
                gt_pts = np.stack([gt_xx, gt_yy], axis=1)

                pd_num = pd_pts.shape[0]
                gt_num = gt_pts.shape[0]

                offset_ths = offset_threshold * ((h**2 + w**2)**0.5)
                distances = np.linalg.norm(
                    np.repeat(pd_pts[:, None, :], repeats=gt_num, axis=1) - \
                    np.repeat(gt_pts[None, :, :], repeats=pd_num, axis=0),
                    axis=2
                )
                precision = np.sum(np.min(distances, axis=1) < offset_ths) / pd_num
                recall = np.sum(np.min(distances, axis=0) < offset_ths) / gt_num
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
        metrics['m_precision'] = round(np.mean(metrics['precision']), 4)
        metrics['m_recall'] = round(np.mean(metrics['recall']), 4)
        metrics['m_f1'] = round(np.mean(metrics['f1']), 4)
        print(f"score_threshold:{score_threshold}, offset_threshold:{offset_threshold}, "
              f"m_precision: {metrics['m_precision']}, m_recall: {metrics['m_recall']}, m_f1: {metrics['m_f1']}")
        return metrics

    def summarize_cum(self, score_threshold=0.5, offset_threshold=0.01, visual_dir=None):
        """
        cntR_total = cntR_total  + cntR ;
        sumR_total  = sumR_total  + sumR ;
        cntP_total  = cntP_total  + cntP ;
        sumP_total  = sumP_total  + sumP ;
        R = cntR_total ./ (sumR_total + (sumR_total ==0));
        P = cntP_total ./ (sumP_total + (sumP_total ==0));
        """
        metrics = {
            'precision': [], 'recall': [], 'f1': [], 'score_threshold': score_threshold,
            'cntR_total': 0, 'sumR_total': 0, 'cntP_total': 0, 'sumP_total': 0
        }
        for gt_target, pred, im_path in self.predictions:
            pred_mask = morphology.skeletonize(
                pred, method='lee').astype(np.uint8)
            h, w = gt_target.shape
            assert (gt_target.shape == pred.shape)
            pred_yy, pred_xx = np.where(pred_mask)
            gt_yy, gt_xx = np.where(gt_target > 0)
            if len(pred_yy) < 1:
                precision, recall, f1 = 0, 0, 0
                cntR, cntP, sumR, sumP = 0, 0, len(gt_yy), 0
            else:
                pd_pts = np.stack([pred_xx, pred_yy], axis=1)
                gt_pts = np.stack([gt_xx, gt_yy], axis=1)
                pd_num = pd_pts.shape[0]
                gt_num = gt_pts.shape[0]
                offset_ths = offset_threshold * ((h**2 + w**2)**0.5)
                distances = np.linalg.norm(
                    np.repeat(pd_pts[:, None, :], repeats=gt_num, axis=1) -
                    np.repeat(gt_pts[None, :, :], repeats=pd_num, axis=0),
                    axis=2)
                cntR = np.sum(np.min(distances, axis=0) < offset_ths)
                cntP = np.sum(np.min(distances, axis=1) < offset_ths)
                sumR = gt_num
                sumP = pd_num
                precision = cntP / sumP
                recall = cntR / sumR
                f1 = get_f1(precision, recall)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['cntR_total'] += cntR
            metrics['sumR_total'] += sumR
            metrics['cntP_total'] += cntP
            metrics['sumP_total'] += sumP
        metrics['m_precision'] = round(np.mean(metrics['precision']), 4)
        metrics['m_recall'] = round(np.mean(metrics['recall']), 4)
        metrics['m_f1'] = round(np.mean(metrics['f1']), 4)
        metrics['cnt_precision'] = round(metrics['cntP_total'] / (metrics['sumP_total'] + (metrics['sumP_total'] == 0)), 4)
        metrics['cnt_recall'] = round(metrics['cntR_total'] / (metrics['sumR_total'] + (metrics['sumR_total'] == 0)), 4)
        metrics['cnt_f1'] = round(get_f1(metrics['cnt_precision'], metrics['cnt_recall']), 4)
        return metrics


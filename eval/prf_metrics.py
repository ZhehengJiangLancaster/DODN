# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Calculate sensitivity and specificity metrics:
 - Precision
 - Recall
 - F-score
"""

import numpy as np
import cv2

from skimage.morphology import skeletonize


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        # print(thresh)
        statistics = []
        statistics_ov = []
        statistics_topo = []

        for pred, gt in zip(pred_list, gt_list):
            gt_img = gt.astype('uint8')
            pred_img = (pred > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
            statistics_ov.append(get_ov_statistics(pred_img, gt_img))
            statistics_topo.append(get_topo_statistics(pred_img, gt_img))

        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        tpm_ov = np.sum([v[0] for v in statistics_ov])
        tpr_ov = np.sum([v[1] for v in statistics_ov])
        fp_ov = np.sum([v[2] for v in statistics_ov])
        fn_ov = np.sum([v[3] for v in statistics_ov])
        tp_topo = np.sum([v[0] for v in statistics_topo])
        pred_sum = np.sum([v[1] for v in statistics_topo])
        gt_sum = np.sum([v[2] for v in statistics_topo])

        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # correctness = 1.0 if tpm_ov==0 and fp_ov==0 else tpm_ov/(tpm_ov+fp_ov)
        correctness = tp_topo / pred_sum
        # calculate recall
        r_acc = tp / (tp + fn)
        # completeness = tpm_ov / (tpm_ov + fn_ov)
        # quality = tpm_ov / (tpm_ov + fn_ov+fp_ov)
        completeness = tp_topo / gt_sum
        quality = tp_topo / (gt_sum + pred_sum - tp_topo)
        new_F_score = 2 * completeness * correctness / (correctness + completeness)
        # calculate f-score
        final_accuracy_all.append(
            [thresh, correctness, completeness, quality, new_F_score, 2 * p_acc * r_acc / (p_acc + r_acc),
             (tpr_ov + tpm_ov) / (tpr_ov + tpm_ov + fp_ov + fn_ov)])
    return final_accuracy_all


def get_ov_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    # filter = np.array([[1,1,1],[1,1,1], [1,1,1]])
    # filter = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    gt = skeletonize(gt).astype('float')
    filter_gt = np.ones((5, 5))
    # gt_tolerance = cv2.filter2D(gt, -1, filter_gt)
    gt_tolerance = cv2.dilate(gt, filter_gt, iterations=1)
    gt_tolerance[gt_tolerance > 0] = 1
    filter_pr = np.ones((5, 5))
    # pr_tolerance = cv2.filter2D(pred, -1, filter_pr)
    # pred = cv2.blur(pred,(5,5))>0.1
    pred = skeletonize(pred).astype('float')
    pr_tolerance = cv2.dilate(pred, filter_pr, iterations=1)
    pr_tolerance[pr_tolerance > 0] = 1
    tpm = np.sum((pred == 1) & (gt_tolerance == 1))
    fp = np.sum((pred == 1) & (gt_tolerance == 0))
    tpr = np.sum((pr_tolerance == 1) & (gt == 1))
    fn = np.sum((pr_tolerance == 0) & (gt == 1))
    return [tpm, tpr, fp, fn]


def get_topo_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    gt = skeletonize(gt).astype('float')
    filter_gt = np.ones((5, 5))
    gt_tolerance = cv2.dilate(gt, filter_gt, iterations=1)
    gt_tolerance[gt_tolerance > 0] = 1
    pred = skeletonize(pred).astype('float')
    tp = np.sum((pred == 1) & (gt_tolerance == 1))
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)
    return [tp, pred_sum, gt_sum]


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    # filter = np.array([[1,1,1],[1,1,1], [1,1,1]])
    filter = np.ones((5, 5))
    gt_tolerance = cv2.filter2D(gt, -1, filter)
    gt_tolerance[gt_tolerance > 0] = 1
    tp = np.sum((pred == 1) & (gt_tolerance == 1))
    fp = np.sum((pred == 1) & (gt_tolerance == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]

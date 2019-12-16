import numpy as np


def point_accuracy(est_labels, gt_labels):
    return np.count_nonzero(est_labels == gt_labels), gt_labels.size


def class_accuracy(est_labels, gt_labels, all_labels=None):
    if all_labels is None:
        all_labels = np.unique([est_labels, gt_labels])
    all_to_gt = all_labels[:, None] == gt_labels[None]
    all_to_est = all_labels[:, None] == est_labels[None]
    return np.count_nonzero(np.logical_and(all_to_gt, all_to_est), axis=1), \
           np.count_nonzero(all_to_gt, axis=1), \
           all_labels


def mean_class_accuracy(class_accuracies, exclude_background=True):
    return np.nanmean(class_accuracies[int(exclude_background):])


def iou(est_labels, gt_labels, all_labels=None):
    if all_labels is None:
        all_labels = np.unique([est_labels, gt_labels])
    gt_eq_label = all_labels[:, None] == gt_labels[None]
    est_eq_label = all_labels[:, None] == est_labels[None]
    intersection = np.count_nonzero(np.logical_and(gt_eq_label, est_eq_label), axis=1)
    union = np.count_nonzero(np.logical_or(gt_eq_label, est_eq_label), axis=1)
    return intersection, union


def miou(iou):
    return np.nanmean(iou)


def fiou(est_labels, gt_labels, all_labels=None):
    if all_labels is None:
        all_labels = np.unique([est_labels, gt_labels])
    gt_eq_label = all_labels[:, None] == gt_labels[None]
    est_eq_label = all_labels[:, None] == est_labels[None]
    t_i = np.count_nonzero(gt_eq_label, axis=1)
    return np.nansum(t_i * np.count_nonzero(np.logical_and(gt_eq_label, est_eq_label), axis=1) /
                     np.count_nonzero(np.logical_or(gt_eq_label, est_eq_label), axis=1)) / t_i.sum()


def precision_recall(est_labels, gt_labels, est_probs, test_probs=None, iou_threshs=None):
    if iou_threshs is None:
        # iou_threshs = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9]
        iou_threshs = [0.5]
    iou_threshs = np.array(iou_threshs)[None]
    if test_probs is None:
        test_probs = list(sorted(np.append(-1e-6, est_probs)))
    all_labels = np.unique([est_labels, gt_labels])
    all_labels = all_labels[all_labels != 0]

    precision = np.empty((len(test_probs), iou_threshs.shape[1]), dtype=np.float)
    recall = np.empty((len(test_probs), iou_threshs.shape[1]), dtype=np.float)

    total_pos_gt = np.count_nonzero(np.unique(gt_labels))
    for prob_ind, prob_thresh in enumerate(test_probs):
        lt_mask = est_probs <= prob_thresh
        est_labels[lt_mask] = 0

        if np.count_nonzero(est_labels) == 0:
            precision[prob_ind:] = 1
            recall[prob_ind:] = 0
            break

        total_pos_est = np.count_nonzero(np.unique(est_labels))
        iou_val = np.nan_to_num(iou(est_labels, gt_labels, all_labels=all_labels))
        tps = np.count_nonzero(iou_val[:, None] >= iou_threshs, axis=0)
        fns = total_pos_gt - tps
        fps = total_pos_est - tps

        precision[prob_ind] = tps / (tps + fps)
        recall[prob_ind] = tps / (tps + fns)

    return precision, recall, test_probs, iou_threshs


# def precision_recall(est_labels, gt_labels, est_probs, test_probs=None):
#     if test_probs is None:
#         test_probs = list(sorted(np.append(-1e-6, est_probs)))
#     all_labels = np.unique([est_labels, gt_labels])
#     all_labels = all_labels[all_labels != 0]
#
#     all_tps = np.empty(len(test_probs), iou_threshs.shape[1]), dtype=np.float)
#     all_fns = np.empty(len(test_probs), iou_threshs.shape[1]), dtype=np.float)
#     all_fps = np.empty(len(test_probs), iou_threshs.shape[1]), dtype=np.float)
#
#     total_pos_gt = np.count_nonzero(np.unique(gt_labels))
#     for prob_ind, prob_thresh in enumerate(test_probs):
#         lt_mask = est_probs <= prob_thresh
#         est_labels[lt_mask] = 0
#
#         if np.count_nonzero(est_labels) == 0:
#             precision[prob_ind:] = 1
#             recall[prob_ind:] = 0
#             break
#
#         total_pos_est = np.count_nonzero(np.unique(est_labels))
#         iou_val = np.nan_to_num(iou(est_labels, gt_labels, all_labels=all_labels))
#         tps = np.count_nonzero(iou_val[:, None] >= iou_threshs, axis=0)
#         fns = total_pos_gt - tps
#         fps = total_pos_est - tps
#
#         precision[prob_ind] = tps / (tps + fps)
#         recall[prob_ind] = tps / (tps + fns)
#
#     return precision, recall, test_probs, iou_threshs


def interpolated_precision(precision):
    """
    Assumes precision is already in reverce order
    :param precision:
    :return:
    """
    result = np.empty(precision.shape, dtype=precision.dtype)
    max_val = -1
    for i, val in enumerate(precision):
        max_val = max(max_val, val)
        result[i] = max_val
    return result
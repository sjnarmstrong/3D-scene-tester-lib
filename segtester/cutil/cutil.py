import numpy as np
def calc_iou_mtx(instance_masks_est, instance_masks_gt):
    masks_p = instance_masks_est[:, None]
    masks_gt = instance_masks_gt[None]
    try:
        intersection = np.count_nonzero(np.bitwise_and(masks_gt, masks_p), axis=2)
        union = np.count_nonzero(np.bitwise_or(masks_gt, masks_p), axis=2)
    except MemoryError:
        intersection = np.zeros((masks_p.shape[0], masks_gt.shape[1]), dtype=np.uint32)
        union = np.zeros((masks_p.shape[0], masks_gt.shape[1]), dtype=np.uint32)
        for i in range(len(intersection)):
            intersection[i] = np.count_nonzero(np.bitwise_and(masks_gt[0], masks_p[i]), axis=1)
            union[i] = np.count_nonzero(np.bitwise_or(masks_gt[0], masks_p[i]), axis=1)
    return intersection/union
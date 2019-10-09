import numpy as np


class Seg:
    def __init__(self, classes, instance_masks, instance_classes, class_map, confidence_scores=None):
        """

        :param classes: (N) -> int class_id
        :param instance_masks: (I, N) -> bool contains
        :param instance_classes: (I) -> int class_id
        :param class_map: (C) -> str class_name
        :param confidence_scores: None | (N) -> float confidence
        """
        self.classes = classes
        self.instance_masks = instance_masks
        self.instance_classes = instance_classes
        self.class_map = class_map
        self.confidence_scores = confidence_scores if confidence_scores is not None else np.ones(len(self.classes))

    def get_instance_ious(self, ground_truth):
        masks_p = self.instance_masks[:, None]
        masks_gt = ground_truth.instance_masks[None]
        intersection = np.count_nonzero(np.bitwise_and(masks_gt, masks_p), axis=2)
        union = np.count_nonzero(np.bitwise_or(masks_gt, masks_p), axis=2)
        return intersection/union

    def get_instance_map(self, ground_truth, min_match_iou=0, allow_duplicates=False,
                         class_map=None, match_classes=False):

        if match_classes and class_map is None:
            class_map = np.arange(self.instance_classes.max()+1)

        iou = self.get_instance_ious(ground_truth)
        instance_map = np.repeat(-1, iou.shape[0])
        unassigned_indices = list(range(iou.shape[0]))
        assigned_labels = {}
        while len(unassigned_indices) > 0:
            i = unassigned_indices.pop()
            max_ind = np.argmax(iou[i])

            iou_val = iou[i, max_ind]
            if iou_val <= min_match_iou:
                continue

            if class_map is not None and ground_truth.instance_classes[max_ind] != class_map[self.instance_classes[i]]:
                unassigned_indices.append(i)
                iou[i, max_ind] = -1
                continue

            if allow_duplicates or max_ind not in assigned_labels:
                instance_map[i] = max_ind
                assigned_labels[max_ind] = i
                continue
            prev_i = assigned_labels[max_ind]
            prev_iou_val = iou[prev_i, max_ind]
            if prev_iou_val >= iou_val:
                unassigned_indices.append(i)
                iou[i, max_ind] = -1
            else:
                unassigned_indices.append(prev_i)
                iou[prev_i, max_ind] = -1
                instance_map[i] = max_ind
                instance_map[prev_i] = -1
                assigned_labels[max_ind] = i
        unassigned_instances = instance_map == -1
        instance_map[unassigned_instances] = \
            np.arange(iou.shape[1], iou.shape[1]+np.count_nonzero(unassigned_instances))
        return instance_map

    def get_segmentation_labels(self, ground_truth, **kwargs):
        labels_gt = np.zeros(ground_truth.classes.size, dtype=np.uint32)
        for i, mask in enumerate(ground_truth.instance_masks):
            labels_gt[mask] = i + 1

        instance_map = self.get_instance_map(ground_truth, **kwargs)
        labels_p = np.zeros(self.classes.size, dtype=np.uint32)
        for i, mask in enumerate(self.instance_masks):
            labels_p[mask.flat] = instance_map[i] + 1
        return labels_p, labels_gt, instance_map

    def get_class_labels(self, ground_truth, class_map=None):
        if class_map is None:
            class_map = np.arange(self.classes.max()+1)
        mapped_labels = class_map[self.classes]
        return mapped_labels.flatten(), ground_truth.classes.flatten()


import numpy as np
from segtester.cutil import cutil
import torch


class Seg:
    def __init__(self, classes, instance_masks, instance_classes, confidence_scores=None):
        """

        :param classes: (N) -> int class_id
        :param instance_masks: (I, N) -> bool contains
        :param instance_classes: (I) -> int class_id
        :param confidence_scores: None | (N) -> float confidence
        """
        self.classes = classes
        self.instance_masks = instance_masks
        self.instance_classes = instance_classes
        self.confidence_scores = confidence_scores if confidence_scores is not None else np.ones(len(self.classes))

    @staticmethod
    def calc_iou_mtx_gpu(a, b):
        a = torch.tensor(a, device='cuda')
        b = torch.tensor(b, device='cuda')
        res = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                int_it = (a[i] * b[j]).sum()
                union_it = (a[i] + b[j]).sum()
                res[i, j] = float(int_it) / float(union_it) if union_it > 0 else 0
        return res

    def get_instance_ious(self, ground_truth):
        if torch.cuda.is_available() and len(self.classes) > 100000:
            try:
                return self.calc_iou_mtx_gpu(self.instance_masks, ground_truth.instance_masks)
            except Exception:
                print("I could not run on gpu reverting to cpu")
                pass

        return cutil.calc_iou_mtx(self.instance_masks, ground_truth.instance_masks)

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
        i = 0
        for i, mask in enumerate(ground_truth.instance_masks):
            labels_gt[mask] = i + 1

        instance_map = self.get_instance_map(ground_truth, **kwargs)
        labels_p = np.zeros(self.classes.size, dtype=np.uint32)
        # for mask_i, unassigned_i in zip(instance_map, unassigned_instances):
        #     if unassigned_i:
        #         labels_p[self.instance_masks[mask_i].flat] = i + 1
        #     else:
        #         i += 1
        #         labels_p[mask_i] = i
        # return labels_p, labels_gt, instance_map
        for i, mask in enumerate(self.instance_masks):
            labels_p[mask.flat] = instance_map[i] + 1

        # from PIL import Image
        # test_img_gt = np.zeros(self.image_shape +(3, ), dtype=np.uint8)
        # test_img_pred = np.zeros(self.image_shape +(3, ), dtype=np.uint8)
        # for i in range(instance_map.max()+1):
        #     test_img_gt[:] = 0
        #     test_img_gt[(labels_gt == i).reshape(self.image_shape[1], self.image_shape[0]).T] = [255, 0, 0]
        #     Image.fromarray(test_img_gt).show()
        #     test_img_pred[:] = 0
        #     test_img_pred[(labels_p == i).reshape(self.image_shape[0], self.image_shape[1])] = [0, 255, 0]
        #     Image.fromarray(test_img_pred).show()

        return labels_p, labels_gt, instance_map

    def get_class_labels(self, ground_truth, class_map=None):
        if class_map is None:
            class_map = np.arange(self.classes.max()+1)
        mapped_labels = class_map[self.classes]
        return mapped_labels.flatten(), ground_truth.classes.flatten()

    def map_own_classes(self, label_map):
        self.classes = label_map[self.classes]
        self.instance_classes = label_map[self.instance_classes]

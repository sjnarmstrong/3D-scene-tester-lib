import numpy as np
from segtester.types.seg import Seg


class Seg2D(Seg):
    def __init__(self, classes, instance_masks, instance_classes, class_map, confidence_scores=None):
        self.image_shape = classes.shape
        super().__init__(
            classes.flatten(),
            instance_masks.reshape((instance_masks.shape[0], -1)),
            instance_classes,
            class_map,
            None if confidence_scores is None else confidence_scores.flatten()
        )

    @staticmethod
    def from_nyuv2_format(filename, indices=None):
        import h5py
        with h5py.File(filename, 'r') as f:
            labels = f.get('labels')
            instances = f.get('instances')
            names = f['names'][0]
            class_map = {i: f[f['names'][0][i]][()].tobytes().decode('utf8') for i in range(len(names))}
            if indices is None:
                indices = range(len(labels))

            res = []
            for i in indices:
                instance_masks, unique_pairs = Seg2D.get_image_masks(labels[i], instances[i])
                res.append(Seg2D(labels[i].T, instance_masks.T, unique_pairs[:, 0], class_map))
            return res

    @staticmethod
    def get_image_masks(label_image, instance_image):
        label_instances = np.concatenate((label_image[:, :, None], instance_image[:, :, None]), axis=2)
        unique_pairs = np.unique(label_instances.reshape(-1, 2), axis=0)
        unique_pairs = unique_pairs[np.sum(unique_pairs, axis=1) > 0]
        return np.equal(label_instances[:, :, None], unique_pairs).all(axis=3), unique_pairs


if __name__ == "__main__":
    MAT_DIR = "/mnt/1C562D12562CEDE8/DATASETS/nyu_depth_v2_labeled.mat"
    segs = Seg2D.from_nyuv2_format(MAT_DIR, [1, 2])
    labels_p, labels_gt, instance_map = segs[0].get_segmentation_labels(segs[1])
    labels_p2, labels_gt2, instance_map2 = segs[0].get_segmentation_labels(segs[1], match_classes=True)
    l1, l2 = segs[0].get_class_labels(segs[1])

    print("Done")

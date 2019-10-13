from segtester.types.seg import Seg
from sklearn.neighbors import BallTree, KDTree
import numpy as np

SEARCH_TREES = {"KD": KDTree, "BALL": BallTree}


class Seg3D(Seg):
    def __init__(self, points, classes, instance_masks, instance_classes, confidence_scores):
        super().__init__(classes, instance_masks, instance_classes, confidence_scores)
        self.points = points
        self.search_tree = None
        if instance_masks is None:
            self.instance_masks, self.instance_classes = self.get_instance_masks_from_classes()

    def get_instance_masks_from_classes(self, dist_thresh=0.60, classes_to_skip=[0]):
        search_tree = self.get_search_tree()
        instance_masks = []
        instance_classes = []
        unassigned_points = np.in1d(self.classes, classes_to_skip) == False
        points_to_search = []
        curr_instance_mask = None
        curr_instance_class = None
        while any(unassigned_points):
            if len(points_to_search) == 0:
                first_unassigned = np.argmax(unassigned_points)
                points_to_search = [self.points[first_unassigned]]
                curr_instance_class = self.classes[first_unassigned]
                instance_classes.append(curr_instance_class)
                curr_instance_mask = np.zeros(len(self.points), dtype=np.bool)
                instance_masks.append(curr_instance_mask)

            nearby_points = search_tree.query_radius(points_to_search, dist_thresh)
            # flattened_ind_it = (nearby_ind for nearby_ind_arr in nearby_points for nearby_ind in nearby_ind_arr)
            points_to_search = []
            for nearby_arr in nearby_points:
                mask = np.logical_and(self.classes[nearby_arr] == curr_instance_class, unassigned_points[nearby_arr])
                nearby_arr = nearby_arr[mask]
                curr_instance_mask[nearby_arr] = True
                unassigned_points[nearby_arr] = False
                points_to_search.extend(self.points[nearby_arr])
        return np.array(instance_masks), np.array(instance_classes)

    def get_search_tree(self, search_tree_type='KD'):
        if self.search_tree is None:
            self.search_tree = SEARCH_TREES[search_tree_type](self.points)
        return self.search_tree

    def get_mapping_to(self, other, k=1, return_distance=True, dualtree=False, breadth_first=False, sort_results=False):
        return other.get_search_tree().query(self.points, k, return_distance, dualtree, breadth_first, sort_results)

    def get_flattened_seg(self, gt_seg, dist_thresh=float('inf')):
        dists, mapping = self.get_mapping_to(gt_seg)
        dists, mapping = dists[:, 0], mapping[:, 0]
        gt_thresh = dists > dist_thresh
        classes = gt_seg.classes[mapping]
        classes[gt_thresh] = 0
        instance_masks = gt_seg.instance_masks[:, mapping]
        instance_masks[:, gt_thresh] = False
        if gt_seg.confidence_scores is not None:
            confidence_scores = gt_seg.confidence_scores[mapping]
            confidence_scores[gt_thresh] = 0
        else:
            confidence_scores = None
        return Seg3D(gt_seg.points[mapping], classes, instance_masks, gt_seg.instance_classes, confidence_scores), dists

    def get_mapped_seg(self, est_seg):
        point_dist, point_mapping = est_seg.get_mapping_to(self)
        point_mapping = point_mapping.ravel()

        classes = self.classes[point_mapping]
        instance_classes = self.instance_classes.copy()
        points = self.points[point_mapping]
        confidence_scores = self.confidence_scores[point_mapping]
        instance_masks = self.instance_masks[:, point_mapping]

        return Seg3D(points, classes, instance_masks, instance_classes, confidence_scores), point_dist

    def get_masked_seg(self, mask):

        classes = self.classes[mask]
        instance_classes = self.instance_classes.copy()
        points = self.points[mask]
        confidence_scores = self.confidence_scores[mask]
        instance_masks = self.instance_masks[:, mask]

        return Seg3D(points, classes, instance_masks, instance_classes, confidence_scores)

    def get_labelled_pcd(self, labels_to_vis=None, max_label=None, point_offset=[0,0,0]):
        import open3d as o3d
        from matplotlib import pyplot as plt
        cmap = plt.get_cmap("hsv")
        if labels_to_vis is None:
            labels_to_vis = self.classes
        max_class = max_label if max_label is not None else labels_to_vis.max()

        vis_cmap = cmap((np.arange(max_class+1)-1)/max_class)
        vis_cmap[0] = (0.3, 0.3, 0.3, 1)

        colors = vis_cmap[labels_to_vis][:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points+point_offset)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def vis_labels(self, labels_to_vis=None, max_label=None, other_pcd=[]):
        import open3d as o3d
        o3d.visualization.draw_geometries([self.get_labelled_pcd(labels_to_vis, max_label)]+other_pcd)

    @staticmethod
    def load_points_and_labels_from_ply(ply_filename, load_conf=True, load_ids=True):
        from plyfile import PlyData
        ply_data = PlyData.read(ply_filename)

        vtx = ply_data['vertex']
        num_verts = vtx.count
        points = np.empty(shape=(num_verts, 3), dtype=np.float64)
        points[:, 0] = vtx.data['x']
        points[:, 1] = vtx.data['y']
        points[:, 2] = vtx.data['z']

        ret = (points,)

        if load_conf:
            conf = np.empty(shape=num_verts, dtype=np.float64)
            try:
                conf[:] = vtx.data['confidence']
            except ValueError:
                conf[:] = 1.0
            ret += (conf,)

        if load_ids:
            ids = ply_data['vertex']['label']
            ret += (ids,)

        return ret

    @staticmethod
    def get_from_scenenn_format(ply_filename, xml_filename, label_map=None):
        from xml.dom import minidom
        import string

        points, conf, ids = Seg3D.load_points_and_labels_from_ply(ply_filename)

        mydoc = minidom.parse(xml_filename)
        items = mydoc.getElementsByTagName('label')

        unique_instance_label_set = set([item.attributes['text'].value for item in items
                                         if item.attributes['text'].value != ''])
        instance_labels = {k: v for v, k in enumerate(unique_instance_label_set, 1)}
        instance_labels[''] = 0

        mapping = {int(item.attributes['id'].value): {
            "text": item.attributes['text'].value,
            "instance_id": instance_labels[item.attributes['text'].value],
            "class": item.attributes['text'].value.rstrip(string.digits)} for item in items}

        labels = np.empty(len(ids), dtype=np.uint16)
        instances = np.empty(len(ids), dtype=np.uint16)
        label_map = {"": 0} if label_map is None else label_map
        current_max_label = -1
        for v in label_map.values():
            current_max_label = max(current_max_label, v)
        for i, id_loop in enumerate(ids):
            instances[i] = mapping[id_loop]["instance_id"]
            class_name = mapping[id_loop]["class"]
            if class_name not in label_map:
                current_max_label += 1
                label_map[class_name] = current_max_label
            labels[i] = label_map[class_name]

        instance_masks = instances[None].astype(np.int) == np.arange(1, len(instance_labels), 1)[:, None]
        mask_labels = [None for _ in instance_labels]
        for v, k in instance_labels.items():
            if v not in label_map:
                current_max_label += 1
                label_map[v] = current_max_label
            mask_labels[k] = label_map[v]
        mask_labels.pop(0)
        return Seg3D(
            points, labels, instance_masks, np.array(mask_labels, dtype=np.uint16), label_map, conf
        )

    @staticmethod
    def get_from_scannet_ply_format(ply_filename, aggregation_filename, seg_filename, label_map=None):
        import json
        points, conf = Seg3D.load_points_and_labels_from_ply(ply_filename, load_ids=False)
        with open(seg_filename) as fp:
            seg_indices = np.array(json.load(fp)['segIndices'])
        with open(aggregation_filename) as fp:
            seg_groups = json.load(fp)['segGroups']
        instance_masks = np.array([np.in1d(seg_indices, seg_group['segments']) for seg_group in seg_groups])
        mask_labels = [seg_group['label'] for seg_group in seg_groups]
        labels = np.repeat(0, len(seg_indices)).astype(np.uint16)

        label_map = {"": 0} if label_map is None else label_map
        current_max_label = -1
        for v in label_map.values():
            current_max_label = max(current_max_label, v)

        encoded_mask_labels = []
        for mask, label in zip(instance_masks, mask_labels):
            if label not in label_map:
                current_max_label += 1
                label_map[label] = current_max_label
            labels[mask] = label_map[label]
            encoded_mask_labels.append(label_map[label])
        return Seg3D(
            points, labels, instance_masks, np.array(encoded_mask_labels, dtype=np.uint16), conf
        )


if __name__ == "__main__":
    data1 = Seg3D.get_from_scannet_ply_format(
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0000_00/scene0000_00_vh_clean_2.ply",
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0000_00/scene0000_00_vh_clean.aggregation.json",
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0000_00/scene0000_00_vh_clean_2.0.010000.segs.json"
    )
    # data2 = Seg3D.get_from_scannet_ply_format(
    #     "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00_vh_clean_2.ply",
    #     "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00_vh_clean.aggregation.json",
    #     "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00_vh_clean_2.0.010000.segs.json"
    # )
    data_gt = Seg3D.get_from_scannet_ply_format(
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0000_00/scene0000_00_vh_clean.ply",
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0000_00/scene0000_00_vh_clean.aggregation.json",
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0000_00/scene0000_00_vh_clean.segs.json"
    )
    from time import time
    t1 = time()
    seg, dists = data1.get_flattened_seg(data_gt)
    t2 = time()
    print(f"Getting aligned seg took {t2-t1}")
    t1 = time()
    l1, l2 = data1.get_class_labels(seg)
    t2 = time()
    print(f"Getting class labels took {t2-t1}")
    print(l1)





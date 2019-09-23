import open3d as o3d
import numpy as np
import json
import os
from segtester import logger
import csv
from segtester.types import Scene, Dataset


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.dataset.scannet import SCANNETConfig


class ScannetScene(Scene):
    def __init__(self, id, info_path, sens_path, mesh_path, segmentation_map, aggregation_map,
                 projected_instance_archive, projected_label_file):
        super().__init__()
        self.id = id
        self.info_path = info_path
        self.sens_path = sens_path
        self.mesh_path = mesh_path
        self.segmentation_map = segmentation_map
        self.aggregation_map = aggregation_map
        self.projected_instance_archive = projected_instance_archive
        self.projected_label_file = projected_label_file

    def get_pcd(self):
        return o3d.io.read_point_cloud(self.mesh_path)

    def get_labeled_pcd(self):
        with open(self.segmentation_map) as fp:
            seg_indices = np.array(json.load(fp)['segIndices'])
        with open(self.aggregation_map) as fp:
            seg_groups = json.load(fp)['segGroups']
        instance_masks = np.array([np.in1d(seg_indices, seg_group['segments']) for seg_group in seg_groups])
        mask_labels = [seg_group['label'] for seg_group in seg_groups]

        return instance_masks, mask_labels, self.get_pcd()


class ScannetDataset(Dataset):

    HD_PATHS = ("{scene_id}.txt", "{scene_id}.sens", "{scene_id}_vh_clean.ply", "{scene_id}_vh_clean.segs.json",
                "{scene_id}.aggregation.json", "{scene_id}_2d-instance-filt.zip",
                "{scene_id}_2d-label-filt.zip")  # using other agg file but either should be fine
    DEDIMATED_PATHS = ("{scene_id}.txt", "{scene_id}.sens", "{scene_id}_vh_clean_2.ply",
                       "{scene_id}_vh_clean_2.0.010000.segs.json", "{scene_id}_vh_clean.aggregation.json",
                       "{scene_id}_2d-instance-filt.zip", "{scene_id}_2d-label-filt.zip")

    def __init__(self, config: 'SCANNETConfig'):
        super().__init__()

        self.base_path = os.path.split(config.file_map)[0]
        self.scenes = []

        with open(config.file_map, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                scene_path = row['SCENE_PATH']
                scene_id = os.path.split(scene_path)[-1]
                use_hd = row['USE_HD']
                scene_paths = ScannetDataset.HD_PATHS if use_hd else ScannetDataset.DEDIMATED_PATHS
                formatted_paths = [f"{self.base_path}/{scene_path}/{pth.format(scene_id=scene_id)}"
                                   for pth in scene_paths]
                scene = ScannetScene(scene_id, *formatted_paths)
                self.scenes.append(scene)

        logger.info(f"Loaded {len(self.scenes)} scenes from {config.file_map}")

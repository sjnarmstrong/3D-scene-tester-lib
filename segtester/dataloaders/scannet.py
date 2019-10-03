import numpy as np
import json
import os
from segtester import logger
import csv
from segtester.types import Scene, Dataset
from segtester.util.sensreader import SensorData
from datetime import datetime
import open3d as o3d


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
        self._sens_data = None

        # For generating mock timestamps as it appears that the timestamps in the sens are missing
        self.start_time = datetime.timestamp(datetime.now())
        self.time_increments = 1 / 60.0

    def get_sens_data(self):
        if self._sens_data is None:
            self._sens_data = SensorData(self.sens_path)
        return self._sens_data

    def get_mock_timestamp(self, iteration_number):
        return self.start_time + iteration_number*self.time_increments

    def get_rgb_depth_image_it(self):
        sens_data = self.get_sens_data()
        for i, image in enumerate(sens_data.get_image_generator()):
            yield image.get_color_image(), image.get_depth_image(), self.get_mock_timestamp(i), i

    def get_depth_position_it(self):
        sens_data = self.get_sens_data()
        for i, image in enumerate(sens_data.get_image_generator()):
            yield image.get_depth_image(), image.camera_to_world, i

    def get_intrinsic_rgb(self):
        return self.get_sens_data().intrinsic_color

    def get_intrinsic_depth(self):
        return self.get_sens_data().intrinsic_depth

    def get_extrinsic_rgb(self):
        return self.get_sens_data().extrinsic_color

    def get_extrinsic_depth(self):
        return self.get_sens_data().extrinsic_depth

    def get_depth_scale(self):
        return self.get_sens_data().depth_shift

    def get_num_frames(self):
        sens_data = self.get_sens_data()
        return sens_data.num_frames

    def get_rgb_size(self):
        return self.get_sens_data().color_height, self.get_sens_data().color_width

    def get_depth_size(self):
        return self.get_sens_data().depth_height, self.get_sens_data().depth_width

    def get_image_info_index(self, index):
        rgbd_image = self.get_sens_data()[index]
        return rgbd_image.get_color_image(), rgbd_image.get_depth_image(), rgbd_image.camera_to_world

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

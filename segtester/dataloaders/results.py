import numpy as np
import os
from segtester import logger
import csv
import open3d as o3d

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.dataset.results import ResultsConfig


class ResultsScene:
    def __init__(self, id, alg_name, base_path, label_map, inv_label_map):
        self.id = id
        self.alg_name = alg_name
        self.base_path = base_path

        self.mesh_path = f"{base_path}/pcd.ply" if os.path.isfile(f"{base_path}/pcd.ply") else None
        self.poses_path = f"{base_path}/poses.npz" if os.path.isfile(f"{base_path}/poses.npz") else None
        self.probs_path = f"{base_path}/probs.npz" if os.path.isfile(f"{base_path}/probs.npz") else None
        frames_dir = f"{base_path}/frames"
        self.frame_paths = [dirit.path for dirit in os.scandir(frames_dir)] if os.path.isdir(frames_dir) else []

        self.cashed_prob_data = None

        self.label_map = label_map
        self.inv_label_map = inv_label_map

    def get_pcd(self):
        return o3d.io.read_point_cloud(self.mesh_path)

    def get_prob_data(self):
        if self.cashed_prob_data is None:
            self.cashed_prob_data = np.load(self.probs_path)
        return self.cashed_prob_data

    def get_probs(self):
        return self.get_prob_data()["likelihoods"]

    def get_instances(self):
        p_dta = self.get_prob_data()
        return None if "instances" not in p_dta else p_dta["instances"]

    def get_labeled_pcd(self, prob_thresh=0):
        probs = self.get_probs()
        labels = np.argmax(probs, axis=1)
        mask_labels = labels[:, None] == np.arange(probs.shape[1])[None]

        return self.get_instances(), mask_labels, self.get_pcd()

    def get_converted_labels(self, mask_labels):
        labels = np.argmax(mask_labels, axis=1)
        return self.label_map[labels]

    def get_pose_path(self):
        from evo.core.trajectory import PosePath3D
        from segtester.types.odometry import Trajectory
        if self.poses_path is None:
            return None
        poses_np = np.load(self.poses_path)["pose_array"]
        poses = [it.reshape(4,4) for it in poses_np]
        return Trajectory(PosePath3D(poses_se3=poses))


class ResultsDataset:
    def __init__(self, config: 'ResultsConfig'):
        import itertools

        super().__init__()

        config_it = config
        base_result_path = None
        while base_result_path is None and config_it is not None:
            if hasattr(config_it, "base_result_path") and isinstance(config_it.base_result_path, str):
                base_result_path = config_it.base_result_path
            config_it = config_it.parent_config
        if base_result_path is None:
            logger.warn("Could not find a base result path from the configs. Attempting to load from current dir.")
            base_result_path = ""

        with open(config.file_map, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            alg_names = next(csv_reader)
            from_label_rows = next(csv_reader)
            to_label_id, to_label_name = next(csv_reader)
            scene_ids = next(csv_reader)

        hold_label_maps = {k: {} for k in alg_names}
        hold_label_names = []
        hold_inv_label_maps = {k: {} for k in alg_names}
        self.max_to_label = 0
        with open(config.label_map, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                hold_label_names.append(row[to_label_name])
                for alg_name, from_label_row in zip(alg_names, from_label_rows):
                    key = int(row[from_label_row]) if row[from_label_row] != '' else 0
                    val = int(row[to_label_id]) if row[to_label_id] != '' and row[from_label_row] != '' else 0
                    self.max_to_label = max(self.max_to_label, val)
                    if row[from_label_row] not in hold_label_maps[alg_name]:  # take the first one matching
                        hold_label_maps[alg_name][key] = val

                    key = int(row[from_label_row]) if row[to_label_id] != '' and row[from_label_row] != '' else 0
                    val = int(row[to_label_id]) if row[from_label_row] != '' else 0
                    if row[to_label_name] not in hold_inv_label_maps[alg_name]:
                        hold_inv_label_maps[alg_name][key] = val

        self.label_maps = {}
        self.label_names = np.array(hold_label_names)
        self.inv_label_maps = {}

        for alg in alg_names:
            hold_from_label_map = np.array(list(hold_label_maps[alg].keys()))
            hold_to_label_map = np.array(list(hold_label_maps[alg].values()))
            label_map = np.zeros(hold_from_label_map.max()+1, dtype=np.uint16)
            label_map[hold_from_label_map] = hold_to_label_map
            self.label_maps[alg] = label_map

            hold_from_label_map = np.array(list(hold_inv_label_maps[alg].keys()))
            hold_to_label_map = np.array(list(hold_inv_label_maps[alg].values()))
            label_map = np.zeros(hold_from_label_map.max()+1, dtype=np.uint16)
            label_map[hold_from_label_map] = hold_to_label_map
            self.inv_label_maps[alg] = label_map

        self.scenes = []
        for alg_name, scene_id in itertools.product(alg_names, scene_ids):
            scene_path = config.safe_format_string(f"{base_result_path}/{config.load_path}",
                                                   dataset_id=config.dataset_id,
                                                   scene_id=scene_id,
                                                   alg_name=alg_name)
            if os.path.isdir(scene_path):
                self.scenes.append(ResultsScene(scene_id, alg_name, scene_path, self.label_maps[alg_name],
                                                self.inv_label_maps[alg_name]))
        logger.debug(f"Successfully found {len(self.scenes)} scenes.")

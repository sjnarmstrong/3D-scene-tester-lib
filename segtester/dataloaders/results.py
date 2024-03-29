import numpy as np
import os
from segtester import logger
import csv
import open3d as o3d
from segtester.types.seg3d import Seg3D
from segtester.types.seg2d import Seg2D

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.dataset.results import ResultsConfig


class ResultsScene:
    def __init__(self, id, alg_name, base_path, label_map_id_col):
        self.id = id
        self.alg_name = alg_name
        self.base_path = base_path

        self.mesh_path = f"{base_path}/pcd.ply" if os.path.isfile(f"{base_path}/pcd.ply") else None
        self.poses_path = f"{base_path}/poses.npz" if os.path.isfile(f"{base_path}/poses.npz") else None
        self.probs_path = f"{base_path}/probs.npz" if os.path.isfile(f"{base_path}/probs.npz") else None
        frames_dir = f"{base_path}/frames"
        self.frame_paths = [dirit.path for dirit in os.scandir(frames_dir)] if os.path.isdir(frames_dir) else []

        self.cashed_prob_data = None

        self.label_map_id_col = label_map_id_col

    def get_seg_2d_from_labeled_frame_id(self, frame_id, output_shape):
        lk = np.load(f"{self.base_path}/frames/frame_{frame_id}.npz")["likelihoods"]
        classes = lk.argmax(axis=2).T
        probs = lk.max(axis=2).T
        v, u = np.meshgrid(np.arange(output_shape[1]), np.arange(output_shape[0]))
        inds = np.stack((u, v))
        inds = np.round(inds * [[[classes.shape[0] - 1]], [[classes.shape[1] - 1]]] /
                        [[[output_shape[0] - 1]], [[output_shape[1] - 1]]]).astype(np.int)
        classes = classes[inds[0], inds[1]]
        unique_classes = np.unique(classes)
        unique_classes = unique_classes[unique_classes!=0]
        instance_masks = classes[None] == unique_classes[:, None, None]
        return Seg2D(classes, instance_masks, unique_classes, probs)

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
        #TODO i think mask labels was intended to be the labels of each mask. Fix me up later...
        assert False
        return self.get_instances(), mask_labels, self.get_pcd()

    def get_pose_path(self):
        from evo.core.trajectory import PosePath3D
        from segtester.types.odometry import Trajectory
        if self.poses_path is None:
            return None
        np_pose_file = np.load(self.poses_path)
        poses_np = np_pose_file["pose_array"].reshape(-1, 4, 4)
        if "init_cam_to_world" in np_pose_file:
            init_cam_to_world = np_pose_file["init_cam_to_world"]
            if (poses_np[0] != init_cam_to_world).any():
                poses_np = init_cam_to_world@poses_np
                #TODO test me properly
        poses = [it.reshape(4, 4) for it in poses_np]
        return Trajectory(PosePath3D(poses_se3=poses))

    def get_seg_3d(self, label_map):
        prob_mtx = self.get_probs()
        labels = np.argmax(prob_mtx, axis=1)
        probs = prob_mtx[np.arange(len(prob_mtx)), labels]
        points = np.array(self.get_pcd().points)

        return Seg3D(
            points, labels, None, None, probs
        )


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
            from_label_cols = next(csv_reader)
            scene_ids = next(csv_reader)
        from_label_cols_dict = {k: v for (k, v) in zip(alg_names, from_label_cols)}

        self.scenes = []
        self.alg_names, self.scene_ids = alg_names, scene_ids
        for alg_name, scene_id in itertools.product(alg_names, scene_ids):
            scene_path = config.safe_format_string(f"{base_result_path}/{config.load_path}",
                                                   dataset_id=config.dataset_id,
                                                   scene_id=scene_id,
                                                   alg_name=alg_name)
            if os.path.isdir(scene_path):
                self.scenes.append(ResultsScene(scene_id, alg_name, scene_path, from_label_cols_dict[alg_name]))
        logger.debug(f"Successfully found {len(self.scenes)} scenes.")

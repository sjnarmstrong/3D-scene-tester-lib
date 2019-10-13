import numpy as np
import json
import os
from segtester import logger
import csv
from segtester.types import Scene, Dataset
from segtester.util.sensreader import SensorData
from datetime import datetime
import open3d as o3d
from segtester.types.seg3d import Seg3D
from segtester.types.seg2d import Seg2D


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.dataset.scannet import SCANNETConfig


class ScannetScene(Scene):
    def __init__(self, id, info_path, sens_path, mesh_path, segmentation_map, aggregation_map,
                 projected_instance_archive, projected_label_file, labelled_pcd_file):
        super().__init__()
        self.id = id
        self.label_map_id_col = "id"
        self.label_map_label_col = "category"
        self.info_path = info_path
        self.sens_path = sens_path
        self.mesh_path = mesh_path
        self.segmentation_map = segmentation_map
        self.aggregation_map = aggregation_map
        self.projected_instance_archive = projected_instance_archive
        self.projected_label_file = projected_label_file
        self.labelled_pcd_file = labelled_pcd_file
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
            yield image.get_color_image(), image.get_depth_image(), image.camera_to_world, self.get_mock_timestamp(i), i

    def get_depth_position_it(self):
        sens_data = self.get_sens_data()
        for i, image in enumerate(sens_data.get_image_generator()):
            yield image.get_depth_image(), image.camera_to_world, i

    def get_rgbd_image_it(self):
        sens_data = self.get_sens_data()
        for i, image in enumerate(sens_data.get_image_generator()):
            yield image, i

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

    def get_pose_path(self):
        from evo.core.trajectory import PosePath3D
        from segtester.types.odometry import Trajectory
        sens_data = self.get_sens_data()
        poses = [image.camera_to_world for image in sens_data.get_image_generator()]
        return Trajectory(PosePath3D(poses_se3=poses))

    def get_labelled_reproj_seg3d(self, depth_min=0.001, depth_max=50.0):
        from zipfile import ZipFile
        import imageio

        d_image_dims = self.get_depth_size()
        rgb_img_dim = self.get_rgb_size()
        # d_image_intrinsics = self.get_intrinsic_depth()
        # rgb_intrinsics = d_image_intrinsics * [[rgb_img_dim[1]], [rgb_img_dim[0]], [1], [1]] / \
        #     [[d_image_dims[1]], [d_image_dims[0]], [1], [1]]
        rgb_intrinsics = self.get_intrinsic_rgb()

        d_scale = self.get_depth_scale()
        inv_intr = np.linalg.inv(rgb_intrinsics)
        v, u, const_1, const_2 = np.meshgrid(
            np.arange(rgb_img_dim[1]),
            np.arange(rgb_img_dim[0]),
            np.array([1]),
            np.array([1]))
        inds = np.stack((v.ravel(), u.ravel(), const_1.ravel(), const_2.ravel()))
        rw_points_cam = inv_intr @ inds
        inds_depth = inds[:2]*[[d_image_dims[1]-1], [d_image_dims[0]-1]]/[[rgb_img_dim[1]-1], [rgb_img_dim[0]-1]]
        inds_depth = np.round(inds_depth).astype(np.int)
        with ZipFile(self.projected_label_file, 'r') as labeled_arch, ZipFile(self.projected_instance_archive, 'r') as instance_arch:
            n_list_lbl = labeled_arch.namelist()
            n_list_inst = instance_arch.namelist()

            for depth_image, camera_to_world, i in self.get_depth_position_it():
                lbl_filt_name, lbl_seg_name = f'label-filt/{i}.png', f'instance-filt/{i}.png'
                if lbl_filt_name not in n_list_lbl or lbl_seg_name not in n_list_inst:
                    continue
                lbl_img = imageio.imread(labeled_arch.open(lbl_filt_name))
                lbl_seg_img = imageio.imread(instance_arch.open(lbl_seg_name))

                scaled_d_img = depth_image[inds_depth[1], inds_depth[0]]/d_scale
                d_img_mask = np.logical_and(scaled_d_img >= depth_min, scaled_d_img <= depth_max)
                proj_points_cam = rw_points_cam[:, d_img_mask]*scaled_d_img[d_img_mask]
                proj_points_cam[3] = 1.0
                proj_points_world = camera_to_world@proj_points_cam
                pt_lbls = lbl_img.flat[d_img_mask]
                pt_seg = lbl_seg_img.flat[d_img_mask]
                stacked_label_inst = np.stack([pt_lbls, pt_seg])
                unique_segs = np.unique(stacked_label_inst, axis=1)
                instance_masks = np.all(np.equal(stacked_label_inst[:, None], unique_segs[:, :, None]), axis=0)
                s3d = Seg3D(
                    proj_points_world.T[:, :3], pt_lbls, instance_masks, unique_segs[0], np.ones(len(pt_lbls))
                )
                pt_lbls = lbl_img.flat
                pt_seg = lbl_seg_img.flat
                stacked_label_inst = np.stack([pt_lbls, pt_seg])
                unique_segs = np.unique(stacked_label_inst, axis=1)
                instance_masks = np.all(np.equal(stacked_label_inst[:, None], unique_segs[:, :, None]), axis=0)
                s2d = Seg2D(
                    lbl_img, instance_masks, unique_segs[0], np.ones(len(pt_lbls))
                )
                yield s3d, s2d, inds[[1, 0]][:, d_img_mask], i

    def get_seg_3d(self, label_map):
        import json
        import difflib
        # points, conf, ids = Seg3D.load_points_and_labels_from_ply(self.labelled_pcd_file)
        points = np.array(self.get_pcd().points)
        with open(self.segmentation_map) as fp:
            seg_indices = np.array(json.load(fp)['segIndices'])
        with open(self.aggregation_map) as fp:
            seg_groups = json.load(fp)['segGroups']
        instance_masks = np.array([np.in1d(seg_indices, seg_group['segments']) for seg_group in seg_groups])
        mask_labels = [seg_group['label'] for seg_group in seg_groups]
        # counts = np.bincount(ids[instance_masks[0]])
        # encoded_mask_labels = [np.argmax(counts)]

        conf = np.ones(len(points))

        text_map = label_map.get_inverse_text_map(self.label_map_id_col, self.label_map_label_col)

        encoded_mask_labels = []
        for lbl in mask_labels:
            if lbl in text_map:
                encoded_mask_labels.append(text_map[lbl])
                continue
            if lbl[-1] == 's' and lbl[:-1] in text_map:
                encoded_mask_labels.append(text_map[lbl[:-1]])
                continue
            lbl_test = lbl.replace(" cabinet", "")
            if lbl_test in text_map:
                encoded_mask_labels.append(text_map[lbl_test])
                continue
            lbl_test = lbl.replace("potted ", "")
            if lbl_test in text_map:
                encoded_mask_labels.append(text_map[lbl_test])
                continue
            closest_matches = difflib.get_close_matches(lbl, list(text_map.keys()), 1)
            assert len(closest_matches) > 0, f"Could not file label: {lbl} in {text_map}"
            encoded_mask_labels.append(text_map[closest_matches[0]])
            logger.warn(f"Could not find lbl: {lbl} assuming it is {closest_matches[0]}")

        ids = np.zeros(instance_masks.shape[1], dtype=np.int)
        for mask, lbl in zip(instance_masks, encoded_mask_labels):
            ids[mask] = lbl

        return Seg3D(
            points, ids, instance_masks, np.array(encoded_mask_labels, dtype=np.int), conf
        )


class ScannetDataset(Dataset):

    HD_PATHS = ("{scene_id}.txt", "{scene_id}.sens", "{scene_id}_vh_clean.ply", "{scene_id}_vh_clean.segs.json",
                "{scene_id}.aggregation.json", "{scene_id}_2d-instance-filt.zip",
                "{scene_id}_2d-label-filt.zip", "{scene_id}_vh_clean_2.labels.ply")
    # using other agg file but either should be fine
    DEDIMATED_PATHS = ("{scene_id}.txt", "{scene_id}.sens", "{scene_id}_vh_clean_2.ply",
                       "{scene_id}_vh_clean_2.0.010000.segs.json", "{scene_id}_vh_clean.aggregation.json",
                       "{scene_id}_2d-instance-filt.zip", "{scene_id}_2d-label-filt.zip",
                       "{scene_id}_vh_clean_2.labels.ply")

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

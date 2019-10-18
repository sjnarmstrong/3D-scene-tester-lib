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
from segtester.util.nyuv2.raw.extract import RawDatasetArchive, RawDatasetScene
from segtester.util.nyuv2.raw.project import depth_rel_to_depth_abs
import h5py


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.dataset.nyuv2 import NYUv2Config


class NYUv2Scene(Scene):
    def __init__(self, _id, raw_scene: RawDatasetScene, gt_files, h5path):
        super().__init__()
        self.id = _id
        self.label_map_id_col = "nyu40id"
        self.label_map_label_col = "nyu40class"
        self.raw_scene = raw_scene
        self.gt_files = gt_files
        self.h5path = h5path

    def get_rgb_depth_image_it(self):
        for frame_ind in range(len(self.raw_scene)):
            np_color_img = self.raw_scene.load_color_image_np(frame_ind)
            np_depth_img = (depth_rel_to_depth_abs(self.raw_scene.load_depth_image_np(frame_ind))*1000)\
                .astype(np.uint16)
            yield np_color_img, np_depth_img, np.eye(4), self.raw_scene.get_timestamp(frame_ind), frame_ind

    def get_depth_position_it(self):
        raise NotImplementedError()

    def get_rgbd_image_it(self):
        raise NotImplementedError()

    def get_intrinsic_rgb(self):
        return np.array([[518.85790117,   0.        , 325.58244941],
                       [  0.        , 519.46961112, 253.73616633],
                       [  0.        ,   0.        ,   1.        ]])

    def get_intrinsic_depth(self):
        return np.array([[582.62448168,   0.        , 313.04475871],
                        [  0.        , 582.69103271, 238.44389627],
                        [  0.        ,   0.        ,   1.        ]])

    def get_extrinsic_rgb(self):
        return np.eye(4)

    def get_extrinsic_depth(self):
        R = np.array([[0.99997799, 0.00505184, 0.00430112],
                      [-0.00503599, 0.99998052, -0.00368798],
                      [-0.00431966, 0.00366624, 0.99998395]])
        T = np.array([0.02503188, 0.00066239, -0.00029342])
        return np.vstack((np.hstack((R,T[:,None])), [[0,0,0,1]]))

    def get_depth_scale(self):
        return 1000.0

    def get_num_frames(self):
        return len(self.raw_scene.frames)

    def get_rgb_size(self):
        return 480, 640

    def get_depth_size(self):
        return 480, 640

    def get_image_info_index(self, index):
        rgbd_image = self.get_sens_data()[index]
        return rgbd_image.get_color_image(), rgbd_image.get_depth_image(), rgbd_image.camera_to_world

    def get_pcd(self):
        raise NotImplementedError()

    def get_labeled_pcd(self):
        raise NotImplementedError()

    def get_pose_path(self):
        raise NotImplementedError()

    def get_labelled_reproj_seg3d(self, depth_min=0.001, depth_max=50.0, predicted_pose=None):
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
        raise NotImplementedError()


class NYUv2Dataset(Dataset):

    def __init__(self, config: 'NYUv2Config'):
        super().__init__()
        self.raw_dataset_archive = RawDatasetArchive(config.zip_file_loc)
        with h5py.File(config.gt_file_loc, 'r') as f:
            depth_frame_names = [f[frame][()].tobytes().decode('utf16') for frame in f.get('rawDepthFilenames')[0]]
            color_frame_names = [f[frame][()].tobytes().decode('utf16') for frame in f.get('rawRgbFilenames')[0]]
            scene_names = [f[frame][()].tobytes().decode('utf16') for frame in f.get('scenes')[0]]
            unique_scenes = set(scene_names)
            gt_scene_name_mapping = {k: [] for k in unique_scenes}
            for scene_name, depth_frame_name, color_frame_name in zip(scene_names, depth_frame_names, color_frame_names):
                gt_scene_name_mapping[scene_name].append((depth_frame_name, color_frame_name))

        # Note some scenes do not seem to have raw datasets. Todo check in parts
        self.scenes = [NYUv2Scene(k, self.raw_dataset_archive[str(k)], gt_scene_name_mapping[k], config.gt_file_loc)
                       for k in unique_scenes if k in self.raw_dataset_archive.scene_frames]

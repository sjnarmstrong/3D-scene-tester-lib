import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

h5path = "/media/sholto/Datasets1/NYUv2/nyu_depth_v2_labeled.mat"
base_results_path = "/media/sholto/Datasets1/results/ActualResults/20191008/NYUv2/"

with h5py.File(h5path, 'r') as f:
    depth_frame_names = [f[frame][()].tobytes().decode('utf16') for frame in f.get('rawDepthFilenames')[0]]
    color_frame_names = [f[frame][()].tobytes().decode('utf16') for frame in f.get('rawRgbFilenames')[0]]
    scene_names = [f[frame][()].tobytes().decode('utf16') for frame in f.get('scenes')[0]]
    depths = f['depths']
    labels = f['labels']
    instances = f['instances']

    v, u, const_1, const_2 = np.meshgrid(
        np.arange(640),
        np.arange(480),
        np.array([1]),
        np.array([1]))
    inds = np.stack((v.ravel(), u.ravel(), const_1.ravel(), const_2.ravel()))


    def get_intrinsic_depth():
        return np.array([[582.62448168, 0., 313.04475871],
                         [0., 582.69103271, 238.44389627],
                         [0., 0., 1.]])


    d_intrinsics = np.eye(4, dtype=np.float64)
    d_intrinsics[:3,:3] = get_intrinsic_depth()

    d_scale = 1000.0
    inv_intr = np.linalg.inv(d_intrinsics)

    inds = np.stack((v.ravel(), u.ravel(), const_1.ravel(), const_2.ravel()))
    rw_points_cam = inv_intr @ inds
    inds_depth = inds[:2] * [[640 - 1], [480 - 1]] / [[640 - 1], [480 - 1]]
    inds_depth = np.round(inds_depth).astype(np.int)

    depth_min=0.001
    depth_max=50.0

    for scene_name, depth_frame_name, depth_image, lbl_img, inst_img in zip(scene_names, depth_frame_names, depths, labels, instances):
        try:
            timestamp_gt = float(depth_frame_name.split('-')[1])
            with open(f'{base_results_path}/{scene_name}/SemanticFusion/elastic_generated.freiburg') as fp:
                odo_data = np.array([[float(it)*1000 for it in line.split()] for line in fp.read().splitlines()])

            best_match_ind = np.argmin(np.abs(odo_data[:, 0]-timestamp_gt))
            best_match_error = np.abs(odo_data[best_match_ind, 0]-timestamp_gt)
            if best_match_error > 0.1:
                print("timestamp match too far off. Skipping frame")
                continue
            camera_to_world = np.eye(4, dtype=np.float64)
            camera_to_world[:3, :3] = R.from_quat(odo_data[best_match_ind, 4:]).as_dcm()
            camera_to_world[0:3, 3] = odo_data[best_match_ind, 1:4]/1000

            scaled_d_img = depth_image[inds_depth[0], inds_depth[1]]
            d_img_mask = np.logical_and(scaled_d_img >= depth_min, scaled_d_img <= depth_max)
            proj_points_cam = rw_points_cam[:, d_img_mask] * scaled_d_img[d_img_mask]
            proj_points_cam[3] = 1.0
            proj_points_world = camera_to_world @ proj_points_cam

            inst_img = inst_img.flatten()[d_img_mask]
            lbl_img = lbl_img.flatten()[d_img_mask]
            stacked_inst_seg = np.stack([inst_img, lbl_img])
            unique_instances = np.unique(stacked_inst_seg, axis=1)
            unique_instances = unique_instances[:, unique_instances[1] != 0]
            inst_mask = stacked_inst_seg[:, None] == unique_instances[:, :, None]
            inst_mask = np.logical_and(inst_mask[0], inst_mask[1])
            loaded_pose = np.load(f'{base_results_path}/{scene_name}/SemanticFusion/poses.npz')['pose_array'][best_match_ind].reshape(4,4)
            print(np.allclose(camera_to_world, loaded_pose))




        except Exception as e:
            print(e)

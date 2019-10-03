import numpy as np
import torch
from tqdm import tqdm
from segtester.types.scene import Scene


def create_image_viewpoints_grid(scene: Scene, vox_dims, occ_start, voxel_size=0.05,
                                 process_nth_frame=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vox_height, vox_width = vox_dims[:2]

    res = torch.zeros(scene.get_num_frames(), vox_width, vox_height, dtype=torch.bool, device=device)

    x, y = np.mgrid[0:scene.get_depth_size()[0], 0:scene.get_depth_size()[1]]

    xyz_cam = np.empty((x.size, 4))
    xyz_cam[:, 0] = y.flatten()
    xyz_cam[:, 1] = x.flatten()
    xyz_cam[:, 2] = 1
    xyz_cam[:, 3] = 1
    xyz_cam = np.linalg.inv(scene.get_intrinsic_depth()) @ xyz_cam.T
    xyz_cam = torch.from_numpy(xyz_cam).to(device)

    depth_scale = scene.get_depth_scale()

    for depth_img, camera_to_world, i in tqdm(scene.get_depth_position_it(), total=scene.get_num_frames(),
                                              desc="Creating viewpoint list"):
        if i % process_nth_frame != 0:
            continue
        depth_img = torch.from_numpy(depth_img.flatten() / depth_scale).to(device)

        xyz_world = xyz_cam * depth_img
        xyz_world[3] = 1
        xyz_world = torch.from_numpy(camera_to_world).to(dtype=torch.double, device=device) @ xyz_world

        inds = torch.round((xyz_world[:2] - occ_start[:2, None]) / voxel_size).to(torch.long)
        inds = inds[:, (inds[0] >= 0) * (inds[1] >= 0) * (inds[0] < vox_width) * (inds[1] < vox_height)]
        res[i][inds[0], inds[1]] = True

    res = res.cpu().numpy()
    all_indicies = []
    for i in range(res.shape[1]):
        row_indicies = []
        for j in range(res.shape[2]):
            row_indicies.append(np.where(res[:, i, j])[0])
        all_indicies.append(row_indicies)
    return all_indicies

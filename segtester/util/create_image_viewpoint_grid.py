import numpy as np
import torch
from tqdm import tqdm
from segtester.types.scene import Scene


def create_image_viewpoints_grid(scene: Scene, vox_dims, min_pcd, voxel_size=0.048, padding=15, process_nth_frame=10):
    vox_height, vox_width = vox_dims

    res = np.zeros((scene.get_num_frames(), vox_width, vox_height), dtype=np.bool)

    x, y = np.mgrid[0:scene.get_depth_size()[0], 0:scene.get_depth_size()[1]]

    xyz_cam = np.empty((x.size, 4))
    xyz_cam[:, 0] = y.flatten()
    xyz_cam[:, 1] = x.flatten()
    xyz_cam[:, 2] = 1
    xyz_cam[:, 3] = 1
    xyz_cam = np.linalg.inv(scene.get_intrinsic_depth()) @ xyz_cam.T
    xyz_cam = torch.from_numpy(xyz_cam).cuda()

    mins_torch = torch.from_numpy(min_pcd[:2, None]).cuda()

    depth_scale = scene.get_depth_scale()

    for depth_img, camera_to_world, i in tqdm(scene.get_depth_position_it(), total=scene.get_num_frames()):
        if i % process_nth_frame != process_nth_frame:
            continue
        depth_img = torch.from_numpy(depth_img.flatten()).cuda() / depth_scale

        xyz_world = xyz_cam * depth_img
        xyz_world[3] = 1
        xyz_world = torch.from_numpy(camera_to_world).cuda().to(torch.double) @ xyz_world

        inds = (torch.round((xyz_world[:2]-mins_torch)/voxel_size) + padding).to(torch.int)
        inds = inds[:, (inds[0] >= 0) * (inds[1] >= 0) * (inds[0] < vox_width) * (inds[1] < vox_height)].cpu().numpy()
        res[i][inds[0], inds[1]] = True

    all_indicies = []
    for i in range(res.shape[1]):
        row_indicies = []
        for j in range(res.shape[2]):
            row_indicies.append(np.where(res[:, i, j])[0])
        all_indicies.append(row_indicies)
    return all_indicies

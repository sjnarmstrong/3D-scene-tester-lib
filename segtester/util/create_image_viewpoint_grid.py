import numpy as np
import torch
from math import ceil
from tqdm import tqdm


def create_grid_of_images(sens_data, max_pcd, min_pcd, voxel_size=0.048, padding=15, process_nth_frame=10):
    vox_width = ceil((max_pcd[0]-min_pcd[0])/voxel_size + 2*padding)
    vox_height = ceil((max_pcd[1]-min_pcd[1])/voxel_size + 2*padding)

    res = np.zeros((sens_data.num_frames, vox_width, vox_height), dtype=np.bool)

    _image = sens_data.get_image_generator().__next__()

    depth_img = _image.get_depth_image() / sens_data.depth_shift
    x, y = np.mgrid[0:depth_img.shape[0], 0:depth_img.shape[1]]

    xyz_cam = np.empty((x.size, 4))
    xyz_cam[:, 0] = y.flatten()
    xyz_cam[:, 1] = x.flatten()
    xyz_cam[:, 2] = 1
    xyz_cam[:, 3] = 1
    xyz_cam = np.linalg.inv(sens_data.intrinsic_depth) @ xyz_cam.T
    xyz_cam = torch.from_numpy(xyz_cam).cuda()

    mins_torch = torch.from_numpy(min_pcd[:2, None]).cuda()

    for i, _image in tqdm(enumerate(sens_data.get_image_generator()), total=sens_data.num_frames):
        if i % process_nth_frame != 0:
            continue
        depth_img = torch.from_numpy((_image.get_depth_image() / sens_data.depth_shift).flatten()).cuda()

        xyz_world = xyz_cam * depth_img.flatten()
        xyz_world[3] = 1
        xyz_world = torch.from_numpy(_image.camera_to_world).cuda().to(torch.double) @ xyz_world

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

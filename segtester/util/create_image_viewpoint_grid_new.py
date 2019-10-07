import open3d as o3d
import torch
from tqdm import tqdm
from itertools import product
import numpy as np

from bisect import bisect_right


def create_image_viewpoints_grid(depth_imgs, camera_to_worlds, vox_dims, occ_start, voxel_size, intrinsic, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    intrinsic = torch.from_numpy(intrinsic).to(dtype=torch.double, device=device)

    img_shape = depth_imgs[0].shape

    vox_height, vox_width = vox_dims[:2]

    xyz_cam = torch.stack(
        torch.meshgrid([torch.arange(0, img_shape[1]),
                        torch.arange(0, img_shape[0]),
                        torch.tensor(1),
                        torch.tensor(1)])
    ).reshape(4, -1).to(dtype=torch.double, device=device)
    xyz_cam[: 2] = (xyz_cam[: 2] - intrinsic[:2, 2, None]) / intrinsic[[0, 1], [0, 1], None]

    res = [[[] for i in range(vox_width)] for j in range(vox_height)]
    keys = [[[] for i in range(vox_width)] for j in range(vox_height)]

    for i, (d_img, camera_to_world) in tqdm(enumerate(zip(depth_imgs, camera_to_worlds)),
                                            desc="Building viewpoints grid",
                                            total=len(depth_imgs)):
        # world_to_camera = torch.inverse(camera_to_world)
        p = xyz_cam * d_img.T.to(dtype=torch.double, device=device).flatten()
        p[3] = 1
        mask = (p[2] >= 0.4) * (p[2] <= 4.0)
        p = camera_to_world.to(dtype=torch.double, device=device) @ p[:, mask]
        inds = torch.round((p[:2] - occ_start[:2, None]) / voxel_size).to(torch.long)
        inds = torch.unique(inds[:2], dim=1)
        inds = inds[:, (inds[0] >= 0) * (inds[1] >= 0) * (inds[0] < vox_height) * (inds[1] < vox_width)].T
        key_val = -len(inds)
        for ind_it in inds:
            keys_it = keys[int(ind_it[0])][int(ind_it[1])]
            insert_index = bisect_right(keys_it, key_val)
            keys_it.insert(insert_index, key_val)
            res[int(ind_it[0])][int(ind_it[1])].insert(insert_index, i)
    return res


def create_image_viewpoints_grid_no_opt(depth_imgs, camera_to_worlds, vox_dims, occ_start, voxel_size, intrinsic, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    intrinsic = torch.from_numpy(intrinsic).to(dtype=torch.double, device=device)

    img_shape = depth_imgs[0].shape

    vox_height, vox_width = vox_dims[:2]

    xyz_cam = torch.stack(
        torch.meshgrid([torch.arange(0, img_shape[1]),
                        torch.arange(0, img_shape[0]),
                        torch.tensor(1),
                        torch.tensor(1)])
    ).reshape(4, -1).to(dtype=torch.double, device=device)
    xyz_cam[: 2] = (xyz_cam[: 2] - intrinsic[:2, 2, None]) / intrinsic[[0, 1], [0, 1], None]

    res = [[[] for i in range(vox_width)] for j in range(vox_height)]

    for i, (d_img, camera_to_world) in tqdm(enumerate(zip(depth_imgs, camera_to_worlds)),
                                            desc="Building viewpoints grid",
                                            total=len(depth_imgs)):
        # world_to_camera = torch.inverse(camera_to_world)
        p = xyz_cam * d_img.T.to(dtype=torch.double, device=device).flatten()
        p[3] = 1
        mask = (p[2] >= 0.4) * (p[2] <= 4.0)
        p = camera_to_world.to(dtype=torch.double, device=device) @ p[:, mask]
        inds = torch.round((p[:2] - occ_start[:2, None]) / voxel_size).to(torch.long)
        inds = torch.unique(inds[:2], dim=1)
        inds = inds[:, (inds[0] >= 0) * (inds[1] >= 0) * (inds[0] < vox_height) * (inds[1] < vox_width)]
        for ind_it in inds.T:
            res[int(ind_it[0])][int(ind_it[1])].append(i)
    return res


def create_image_viewpoints_grid_older(depth_imgs, camera_to_worlds, vox_dims, occ_start, voxel_size, intrinsic, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_shape = depth_imgs[0].shape

    vox_height, vox_width = vox_dims[:2]

    xyz_cam = torch.stack(
        torch.meshgrid([torch.arange(0, img_shape[1]),
                        torch.arange(0, img_shape[0]),
                        torch.tensor(1),
                        torch.tensor(1)])
    ).reshape(4, -1).to(dtype=torch.double, device=device)
    xyz_cam[: 2] = (xyz_cam[: 2] - intrinsic[:2, 2, None]) / intrinsic[[0, 1], [0, 1], None]

    res = [[[] for i in range(vox_width)] for j in range(vox_height)]

    for i, (d_img, camera_to_world) in tqdm(enumerate(zip(depth_imgs, camera_to_worlds)),
                                            desc="Building viewpoints grid",
                                            total=len(depth_imgs)):
        # world_to_camera = torch.inverse(camera_to_world)
        p = camera_to_world.to(dtype=torch.double, device=device) @ xyz_cam
        inds = torch.round((p[:2] - occ_start[:2, None]) / voxel_size).to(torch.long)
        inds = inds[:, (inds[0] >= 0) * (inds[1] >= 0) * (inds[0] < vox_height) * (inds[1] < vox_width)]
        min_inds = inds.min(dim=1)[0]
        max_inds = inds.max(dim=1)[0].int()
        for indx, indy in product(range(int(min_inds[0]), int(max_inds[0])),
                                  range(int(min_inds[1]), int(max_inds[1]))):
            res[indx][indy].append(i)
    return res


def create_image_viewpoints_grid_old(depth_min, depth_max, img_shape, camera_to_worlds, vox_dims, occ_start, voxel_size,
                                 intrinsic, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vox_height, vox_width, vox_depth = vox_dims

    ind_vox = torch.stack(
        torch.meshgrid([torch.arange(0, vox_height),
                        torch.arange(0, vox_width),
                        torch.arange(0, vox_depth),
                        torch.tensor(1)])
    ).reshape(4, -1).to(dtype=torch.double, device=device)

    xyz_vox = ind_vox*voxel_size
    xyz_vox[3] = 1
    xyz_vox[:3] += occ_start[:, None]

    xyz_cam = torch.from_numpy(np.array([[0, 0], [img_shape[1], img_shape[0]]]).T).to(dtype=torch.double, device=device)

    xyz_cam = (xyz_cam - intrinsic[:2, 2, None]) / intrinsic[[0, 1], [0, 1], None]
    min_bounds = xyz_cam.min(dim=1)[0]
    max_bounds = xyz_cam.max(dim=1)[0]

    res = [[[] for i in range(vox_width)] for j in range(vox_height)]

    for i, camera_to_world in tqdm(enumerate(camera_to_worlds),
                                   desc="Building viewpoints grid",
                                   total=len(camera_to_worlds)):
        world_to_camera = torch.inverse(camera_to_world)
        p = world_to_camera.to(dtype=torch.double, device=device) @ xyz_vox
        mask = (p[2] >= depth_min) * (p[2] <= depth_max)
        p = p/p[2]
        mask *= (p[0] >= min_bounds[0]) * (p[1] >= min_bounds[1]) * (p[0] < max_bounds[0]) * (p[1] < max_bounds[1])
        inds = torch.unique(ind_vox[:2, mask], dim=1)
        inds = inds[:, (inds[0] >= 0) * (inds[1] >= 0) * (inds[0] < vox_height) * (inds[1] < vox_width)]
        for ind_it in inds.T:
            res[int(ind_it[0])][int(ind_it[1])].append(i)

    return res


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def visualise_viewpoints(viewpoints, scene_occ, occ_start, colour_imgs):
    from torchvision import transforms

    inverse_transform = transforms.Compose([
        UnNormalize([0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129]),
        transforms.ToPILImage()
    ])

    coords = torch.stack(
        torch.meshgrid([torch.arange(0, scene_occ.shape[0]),
                        torch.arange(0, scene_occ.shape[1]),
                        torch.arange(0, scene_occ.shape[2])])
    ).permute((0, 3, 1, 2)).reshape(3, -1).to(dtype=torch.double)

    coords_scaled = coords*0.05 + occ_start[:, None].cpu()
    occ = scene_occ.permute((2,0,1))
    coords_scaled = coords_scaled[:, occ.reshape(-1)]
    coords = coords[:, occ.reshape(-1)]

    pred_pcd = o3d.geometry.PointCloud()
    points = coords_scaled.numpy().T
    pred_pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pred_pcd])
    colors = np.empty_like(points)

    coords = coords.numpy().astype(np.int)
    for i, img in enumerate(colour_imgs):
        colors[:] = [0.35, 0.35, 0.35]
        t_img = inverse_transform(img)
        t_img.show()
        for x, y in product(range(scene_occ.shape[0]),range(scene_occ.shape[1])):
            if i in viewpoints[x][y]:
                colors[(coords[0] == x) * (coords[1] == y)] = [1.0, 0, 0]

        pred_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pred_pcd])

if __name__ == "__main__":
    from torchvision import transforms



    scene_occ = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0000_00/3DMV/torch/scene_occ.torch")
    colour_imgs = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0000_00/3DMV/torch/color_images.torch")
    depth_imgs = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0000_00/3DMV/torch/depth_images.torch")
    poses = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0000_00/3DMV/torch/poses.torch")
    occ_start = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0000_00/3DMV/torch/occ_start.torch")
    intrinsic = torch.load("/home/sholto/Desktop/test_data/intrinsic.torch")

    res = create_image_viewpoints_grid(0.4, 4, depth_imgs[0].shape, poses, (scene_occ.shape[2], scene_occ.shape[3], scene_occ.shape[1]), occ_start, 0.05, intrinsic)
    # res = create_image_viewpoints_grid(depth_imgs, poses, (scene_occ.shape[2], scene_occ.shape[3], scene_occ.shape[1]), occ_start, 0.05, intrinsic)
    visualise_viewpoints(res, (scene_occ[0]*scene_occ[1]).permute((1,2,0)), occ_start, colour_imgs)



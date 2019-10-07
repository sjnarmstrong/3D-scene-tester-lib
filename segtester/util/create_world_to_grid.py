import open3d as o3d
import torch


def create_world_to_grids(occ_grid_shape, occ_start, voxel_size=0.05, padding_x=15, padding_y=15, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    worlds_to_grid = torch.zeros(occ_grid_shape[:2] + (4, 4), dtype=torch.float, device=device)
    scale = 1 / voxel_size
    worlds_to_grid[:, :, 0, 0] = scale
    worlds_to_grid[:, :, 1, 1] = scale
    worlds_to_grid[:, :, 2, 2] = scale
    worlds_to_grid[:, :, 3, 3] = 1

    # worlds_to_grid[:, :, 0, 3] = -(torch.arange(occ_grid_shape[0],
    #                                             dtype=torch.float,
    #                                             device=device)[:, None] + occ_start[0] * scale- padding_x-1)
    # worlds_to_grid[:, :, 1, 3] = -(torch.arange(occ_grid_shape[1],
    #                                             dtype=torch.float,
    #                                             device=device)[None] + occ_start[1] * scale- padding_y-1)
    worlds_to_grid[:, :, 0, 3] = -(torch.arange(occ_grid_shape[0],
                                                dtype=torch.float,
                                                device=device)[:, None] + occ_start[0] * scale- padding_x)
    worlds_to_grid[:, :, 1, 3] = -(torch.arange(occ_grid_shape[1],
                                                dtype=torch.float,
                                                device=device)[None] + occ_start[1] * scale- padding_y)
    worlds_to_grid[:, :, 2, 3] = -occ_start[2] * scale

    return worlds_to_grid


if __name__ == "__main__":
    from torchvision import transforms
    import open3d as o3d


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

    scene_occ = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0707_00/3DMV/torch/scene_occ.torch")
    colour_imgs = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0707_00/3DMV/torch/color_images.torch")
    depth_imgs = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0707_00/3DMV/torch/depth_images.torch")
    poses = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0707_00/3DMV/torch/poses.torch")
    occ_start = torch.load("/mnt/1C562D12562CEDE8/RESULTS/TestNoDate/SCANNET/scene0707_00/3DMV/torch/occ_start.torch")
    intrinsic = torch.load("/home/sholto/Desktop/test_data/intrinsic.torch")

    scene_occ_sz = scene_occ.shape[1:]

    world_to_grids = create_world_to_grids(scene_occ.shape[2:], occ_start, voxel_size=0.05)

    yx_iter = ((y, x) for y in range(15, scene_occ_sz[1] - 15)
               for x in range(15, scene_occ_sz[2] - 15))
    for (y, x) in yx_iter:
        if x %10 !=0 or y %10 !=0:
            continue
        w_t_g = world_to_grids[y, x]
        grid_to_world = torch.inverse(w_t_g).cpu()


        lin_ind_volume = torch.arange(0, 31*31*62, out=torch.LongTensor())
        coords = poses[0].new(4, lin_ind_volume.size(0)) # ment to construct new tensor with same values as current tensor
        coords[2] = lin_ind_volume / (31*31)
        tmp = lin_ind_volume - (coords[2]*31*31).long()
        coords[1] = tmp / 31
        coords[0] = torch.remainder(tmp, 31)
        coords[3].fill_(1)




        inverse_transform = transforms.Compose([
            UnNormalize([0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129]),
            transforms.ToPILImage()
        ])
        test = (grid_to_world@coords).numpy()
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(test[:3].T)
        pcd3 = o3d.io.read_point_cloud(
            "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans_test/scene0707_00/scene0707_00_vh_clean_2.ply")
        o3d.visualization.draw_geometries([pred_pcd, pcd3])
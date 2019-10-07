import torch


def create_pcd_from_occ(occ_grid, occ_start, voxel_size=0.05, device=None, permute_shape=(2, 0, 1)):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coords = torch.stack(
        torch.meshgrid([torch.arange(0, occ_grid.shape[0]),
                        torch.arange(0, occ_grid.shape[1]),
                        torch.arange(0, occ_grid.shape[2])])
    ).permute((0,) + tuple((p+1 for p in permute_shape))).reshape(3, -1).to(dtype=torch.double, device=device)
    coords *= voxel_size
    coords += occ_start[:, None]
    coords = coords[:, occ_grid.permute(permute_shape).reshape(-1)]

    return coords
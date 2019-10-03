import torch


def create_world_to_grids(occ_grid_shape, occ_start, voxel_size=0.05, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    worlds_to_grid = torch.zeros(occ_grid_shape[:2] + (4, 4), dtype=torch.float, device=device)
    scale = 1 / voxel_size
    worlds_to_grid[:, :, 0, 0] = scale
    worlds_to_grid[:, :, 1, 1] = scale
    worlds_to_grid[:, :, 2, 2] = scale
    worlds_to_grid[:, :, 3, 3] = 1

    worlds_to_grid[:, :, 0, 3] = -(torch.arange(occ_grid_shape[0],
                                                dtype=torch.float,
                                                device=device)[:, None] + occ_start[0] * scale)
    worlds_to_grid[:, :, 1, 3] = -(torch.arange(occ_grid_shape[1],
                                                dtype=torch.float,
                                                device=device)[None] + occ_start[1] * scale)
    worlds_to_grid[:, :, 2, 3] = -occ_start[2] * scale

    return worlds_to_grid

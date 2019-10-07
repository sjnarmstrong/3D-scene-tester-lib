import torch
from segtester.types.scene import Scene


def create_tensor_occ(scene: Scene, voxel_size=0.05, padding_x=31 // 2, padding_y=31 // 2, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coords = torch.tensor(scene.get_pcd().points).to(device)

    min_coord, max_coord = torch.min(coords, dim=0)[0], torch.max(coords, dim=0)[0]

    occ_size_float = (max_coord - min_coord) / voxel_size
    occ_start = min_coord - (voxel_size * occ_size_float.remainder(1.0) / 2.0)
    occ_start[0] -= (padding_x+1) * voxel_size
    occ_start[1] -= (padding_y+1) * voxel_size
    occ_size = occ_size_float.ceil()
    occ_size[0] += 2 * (padding_x+1)
    occ_size[1] += 2 * (padding_y+1)
    occ_size[2] += 1

    occ_grid = torch.zeros(int(occ_size[0]), int(occ_size[1]), int(occ_size[2]), dtype=torch.bool, device=device)

    grid_coords = torch.round((coords - occ_start) / voxel_size).to(torch.long)
    grid_coords = torch.unique(grid_coords.T, dim=1)
    occ_grid[grid_coords[0], grid_coords[1], grid_coords[2]] = True

    return occ_grid, occ_start


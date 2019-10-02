# The goal is to create a way to produce a matrix "world_to_grid" such that can be used to translate world points
# to grid points in a similar fashion as:
# "pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))"
# here we declare the grid as all the indicies within the 31x31x62 portion
import open3d as o3d
import torch
pcd = o3d.io.read_point_cloud("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans_test/scene0707_00/scene0707_00_vh_clean_2.ply")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
voxel_size = 0.05
padding = 31//2
# max_height = 62
# center_pcd_height = False


#define get occ grid

coords = torch.tensor(pcd.points).to(device)

min_coord, max_coord = torch.min(coords, dim=0)[0], torch.max(coords, dim=0)[0]

# Bellow tries to center the points to the center of a voxel. This isnt the cas though i dont feel.
# mincoord can be on the edge of a voxel
# occ_size_no_pad = ((max_coord+voxel_size/2) - (min_coord-voxel_size/2)) / voxel_size
# occ_size_no_pad = (max_coord+voxel_size/2 - min_coord+voxel_size/2) / voxel_size
# occ_size_no_pad = (max_coord+voxel_size - min_coord) / voxel_size
occ_size_float = (max_coord - min_coord) / voxel_size
occ_start = min_coord - (occ_size_float.remainder(1.0) / 2.0)
occ_start[:2] -= padding * voxel_size
occ_size = occ_size_float.ceil()
occ_size[:2] += 2*padding

occ_grid = torch.zeros(int(occ_size[0]), int(occ_size[1]), int(occ_size[2]), dtype=torch.bool, device=device)

grid_coords = torch.round((coords-occ_start)/voxel_size).to(torch.long)
occ_grid[grid_coords] = True


# ddefine get world_to grids
worlds_to_grid = torch.zeros(occ_grid.shape[:2]+(4, 4), dtype=torch.float, device=device)
scale = 1/voxel_size
worlds_to_grid[:, :, 0, 0] = scale
worlds_to_grid[:, :, 1, 1] = scale
worlds_to_grid[:, :, 2, 2] = scale
worlds_to_grid[:, :, 3, 3] = 1

worlds_to_grid[:, :, 0, 3] = -(torch.arange(occ_grid.shape[0],
                                            dtype=torch.float,
                                            device=device)[:, None]+occ_start[0]*scale)
worlds_to_grid[:, :, 1, 3] = -(torch.arange(occ_grid.shape[1],
                                            dtype=torch.float,
                                            device=device)[None]+occ_start[1]*scale)
worlds_to_grid[:, :, 2, 3] = -occ_start[2]*scale


test = worlds_to_grid[1, 1] @ torch.tensor([0.4468, 1.1279, 0.4468, 1],
                                           dtype=torch.float,
                                           device=device)

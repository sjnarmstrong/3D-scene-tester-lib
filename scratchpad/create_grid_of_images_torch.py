from segtester.util.sensreader import SensorData
import open3d as o3d
import numpy as np


def points_to_voxel_inds(xyz, mins, voxel_size, padding):
    return (np.round((xyz-mins)/voxel_size) + padding).astype(np.int)

if __name__ == "__main__":
    from tqdm import tqdm
    from math import ceil
    _sdata = SensorData("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00.sens")
    print(_sdata)
    pcd = o3d.io.read_point_cloud(
        "/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00_vh_clean_2.ply")


    voxel_size = 0.048
    padding = 31 // 2
    pcd_np = np.array(pcd.points)
    min_pcd = np.min(pcd_np, axis=0)
    max_pcd = np.max(pcd_np, axis=0)
    range_pcd = min_pcd - max_pcd

    vox_width = ceil((max_pcd[0]-min_pcd[0])/voxel_size + 2*padding)
    vox_height = ceil((max_pcd[1]-min_pcd[1])/voxel_size + 2*padding)

    res = np.zeros((_sdata.num_frames, vox_width, vox_height), dtype=np.bool)

    _image = _sdata.get_image_generator().__next__()

    depth_img = _image.get_depth_image() / _sdata.depth_shift
    x, y = np.mgrid[0:depth_img.shape[0], 0:depth_img.shape[1]]

    xyz_cam = np.empty((x.size, 4))
    xyz_cam[:, 0] = y.flatten()
    xyz_cam[:, 1] = x.flatten()
    xyz_cam[:, 2] = 1
    xyz_cam[:, 3] = 1
    xyz_cam = np.linalg.inv(_sdata.intrinsic_depth) @ xyz_cam.T

    for i, _image in tqdm(enumerate(_sdata.get_image_generator()), total=_sdata.num_frames):
        if i % 1 !=0:
            continue
        depth_img = (_image.get_depth_image() / _sdata.depth_shift).flatten()

        xyz_world = xyz_cam.copy()
        xyz_world[:2] *= depth_img.flatten()
        xyz_world = _image.camera_to_world @ xyz_world

        inds = points_to_voxel_inds(xyz_world[:2], min_pcd[:2, None], voxel_size, padding)
        inds = inds[:, np.logical_and(np.logical_and(inds[0] >= 0, inds[1] >= 0),
                                      np.logical_and(inds[0] < vox_width, inds[1] < vox_height))]
        res[i][inds[0], inds[1]] = True

    all_indicies = []
    for i in range(res.shape[1]):
        row_indicies = []
        for j in range(res.shape[2]):
            row_indicies.append(np.where(res[:, i, j])[0])
        all_indicies.append(row_indicies)
    print(all_indicies)

from segtester.util.sensreader import SensorData
import open3d as o3d
import numpy as np


def points_to_voxel_inds(xyz, mins, voxel_size, padding):
    return (np.round((xyz-mins)/voxel_size) + padding).astype(np.int)

if __name__ == "__main__":
    from PIL import Image
    from time import sleep
    from tqdm import tqdm
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

    vox_coords = np.mgrid[min_pcd[0] - voxel_size * padding:max_pcd[0] + voxel_size * padding:voxel_size,
                          min_pcd[1] - voxel_size * padding:max_pcd[1] + voxel_size * padding:voxel_size]

    res = np.zeros((_sdata.num_frames, ) + vox_coords.shape[1:3], dtype=np.bool)

    for i, _image in tqdm(enumerate(_sdata.get_image_generator()), total=_sdata.num_frames):
        if i % 1 !=0:
            continue
        color_img = _image.get_color_image()
        x, y = np.mgrid[0:color_img.shape[0], 0:color_img.shape[1]]
        depth_img = _image.get_depth_image() / _sdata.depth_shift
        x_depth = np.clip(np.round(x*depth_img.shape[0]/color_img.shape[0]), 0, depth_img.shape[0]-1).astype(np.int)
        y_depth = np.clip(np.round(y*depth_img.shape[1]/color_img.shape[1]), 0, depth_img.shape[1]-1).astype(np.int)
        depth_img = depth_img[x_depth, y_depth]
        inv_int_depth = np.linalg.inv(_sdata.intrinsic_depth)

        XYZ = np.empty((x.size, 4))
        XYZ[:, 0] = y_depth.flatten()
        XYZ[:, 1] = x_depth.flatten()
        XYZ[:, 2] = 1
        XYZ[:, 3] = 1

        XYZ = inv_int_depth @ XYZ.T
        XYZ[0] *= depth_img.flatten()
        XYZ[1] *= depth_img.flatten()
        XYZ[2] *= depth_img.flatten()
        XYZ = _image.camera_to_world @ XYZ

        inds = points_to_voxel_inds(XYZ[:2], min_pcd[:2, None], voxel_size, padding)
        inds = inds[:, np.logical_and(np.logical_and(inds[0] >= 0, inds[1] >= 0),
                                      np.logical_and(inds[0] < res.shape[1], inds[1] < res.shape[2]))]
        res[i][inds[0], inds[1]] = True

    all_indicies = []
    for i in range(res.shape[1]):
        row_indicies = []
        for j in range(res.shape[2]):
            row_indicies.append(np.where(res[:, i, j]))
        all_indicies.append(row_indicies)
    print(all_indicies)

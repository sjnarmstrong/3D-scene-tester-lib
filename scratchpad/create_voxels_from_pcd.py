import open3d as o3d
import os
import struct
import numpy as np

from sklearn.neighbors import KDTree


def load_scene(filename, num_classes, load_gt):
    assert os.path.isfile(filename)
    fin = open(filename, 'rb')
    # read header
    width = struct.unpack('<I', fin.read(4))[0]
    height = struct.unpack('<I', fin.read(4))[0]
    depth = struct.unpack('<I', fin.read(4))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]

    numElems = width * height * depth
    sdfs = struct.unpack('f'*numElems, fin.read(numElems*4))  #grid3<float>
    labels = None
    if load_gt:
        labels = struct.unpack('B'*numElems, fin.read(numElems))  #grid3<uchar>
    fin.close()
    sdfs = np.asarray(sdfs, dtype=np.float32).reshape([depth, height, width])
    if load_gt:
        labels = np.asarray(labels, dtype=np.uint8).reshape([depth, height, width])
    occ = np.ndarray((2, depth, height, width), np.dtype('B')) #occupancy grid for occupied/empty space, known/unknown space
    occ[0] = np.less_equal(np.abs(sdfs), 1)
    occ[1] = np.greater_equal(sdfs, -1)
    if load_gt:
        # ensure occupied space has non-zero labels
        labels[np.logical_and(np.equal(occ[0], 1), np.equal(labels, 0))] = num_classes - 1
        # ensure non-occupied space has zero labels
        labels[np.equal(occ[0], 0)] = 0
        labels[np.greater_equal(labels, num_classes)] = num_classes - 1
    return occ, labels


occ, labels = load_scene("/mnt/1C562D12562CEDE8/DATASETS/3DMV/scenenn_test/scene0707_00.sdf.ann", 42, False)
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = np.where(occ)

pcd = o3d.io.read_point_cloud("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans_test/scene0707_00/scene0707_00_vh_clean_2.ply")

voxel_size = 0.048
padding = 31//2
pcd_np = np.array(pcd.points)
min_pcd = np.min(pcd_np, axis=0)
max_pcd = np.max(pcd_np, axis=0)
range_pcd = min_pcd - max_pcd

vox_coords = np.mgrid[min_pcd[0]-voxel_size*padding:max_pcd[0]+voxel_size*padding:voxel_size,
                      min_pcd[1]-voxel_size*padding:max_pcd[1]+voxel_size*padding:voxel_size,
                      min_pcd[2]:max_pcd[2]:voxel_size]

dists, mapping = KDTree(pcd_np).query(vox_coords.reshape(3, -1).T, 1, True, False, False, False)

color_hold = np.array(pcd.colors)[mapping[:,0]]
color_hold = color_hold[np.logical_and(np.less_equal(np.abs(dists[:,0]), 0.048), np.less_equal(dists[:,0], 0.1))]

dists = np.swapaxes(dists.reshape(vox_coords.shape[1:]), -1, 0)

occ2 = np.ndarray((2, vox_coords.shape[3], vox_coords.shape[2], vox_coords.shape[1]),
                 np.dtype('B'))  # occupancy grid for occupied/empty space, known/unknown space
occ2[0] = np.less_equal(np.abs(dists), 0.048)
occ2[1] = np.less_equal(dists, 0.1)

pcd_loaded = o3d.geometry.PointCloud()
pcd_loaded.points = o3d.utility.Vector3dVector(np.array(np.where(occ[0]*occ[1])).T)
pcd_gen = o3d.geometry.PointCloud()
pcd_gen.points = o3d.utility.Vector3dVector(np.array(np.where(occ2[0]*occ2[1])).T)
pcd_gen.colors = o3d.utility.Vector3dVector(color_hold)
o3d.visualization.draw_geometries([pcd_loaded, pcd_gen])

# downpcd = pcd.voxel_down_sample(voxel_size=0.01)
# num_true = np.count_nonzero(occ[0]*occ[1])
# pcd_np_1 = np.array(downpcd.points)
#
# pcd2 = o3d.geometry.PointCloud()
# pcd_np = np.array(np.where(occ[0]*occ[1])).T*0.048
# pcd_np[:, [0, 1, 2]] = pcd_np[:, [2, 1, 0]]
#
# top_left_2 =pcd_np[np.argmax(pcd_np[:,0]**2+pcd_np[:,1]**2+pcd_np[:,1]**2)]
# top_left_1 =pcd_np_1[np.argmax(pcd_np_1[:,0]**2+pcd_np_1[:,1]**2+pcd_np_1[:,1]**2)]
#
# pcd_np = pcd_np + top_left_1 - top_left_2
#
# pcd2.points = o3d.utility.Vector3dVector(pcd_np)
# o3d.visualization.draw_geometries([downpcd, pcd2])
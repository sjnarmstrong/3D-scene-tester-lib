from segtester.util.sensreader import SensorData
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    from PIL import Image
    from time import sleep
    _sdata = SensorData("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00.sens")
    print(_sdata)
    outXYZ = np.empty((0, 3), dtype=np.float)
    outColours = np.empty((0, 3), dtype=np.float)
    for i, _image in enumerate(_sdata.get_image_generator()):
        if i % 100 !=0:
            continue
        color_img = _image.get_color_image()
        x, y = np.mgrid[0:color_img.shape[0], 0:color_img.shape[1]]
        depth_img = _image.get_depth_image() / _sdata.depth_shift # not sure if thats right
        x_depth = np.clip(np.round(x*depth_img.shape[0]/color_img.shape[0]), 0, depth_img.shape[0]-1).astype(np.int)
        y_depth = np.clip(np.round(y*depth_img.shape[1]/color_img.shape[1]), 0, depth_img.shape[1]-1).astype(np.int)
        depth_img = depth_img[x_depth, y_depth]

        inv_int_depth = np.linalg.inv(_sdata.intrinsic_depth)
        world_to_cam = _image.camera_to_world
        XYZ = np.empty((x.size, 4))
        XYZ[:, 0] = y_depth.flatten()
        XYZ[:, 1] = x_depth.flatten()
        XYZ[:, 2] = 1
        XYZ[:, 3] = 1

        XYZ = inv_int_depth @ XYZ.T
        XYZ[0] *= depth_img.flatten()
        XYZ[1] *= depth_img.flatten()
        XYZ[2] *= depth_img.flatten()
        XYZ = world_to_cam @ XYZ
        outXYZ = np.vstack((outXYZ, XYZ[:3].T))
        outColours = np.vstack((outColours, color_img.reshape(-1, 3)/255))


    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(outXYZ)
    pred_pcd.colors = o3d.utility.Vector3dVector(outColours)
    o3d.visualization.draw_geometries([pred_pcd])


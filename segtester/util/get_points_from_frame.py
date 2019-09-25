from segtester.util.sensreader import SensorData
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    from PIL import Image
    from time import sleep
    _sdata = SensorData("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00.sens")
    print(_sdata)
    for i, _image in enumerate(_sdata.get_image_generator()):
        if i % 100 !=0:
            continue
        color_img = _image.get_color_image()
        x, y = np.mgrid[0:color_img.shape[0], 0:color_img.shape[1]]
        depth_img = _image.get_depth_image() / _sdata.depth_shift # not sure if thats right
        x_depth = np.clip(np.round(x*depth_img.shape[0]/color_img.shape[0]),0,depth_img.shape[0]-1).astype(np.int)
        y_depth = np.clip(np.round(y*depth_img.shape[1]/color_img.shape[1]),0,depth_img.shape[1]-1).astype(np.int)
        depth_img = depth_img[x_depth, y_depth]
        inv_int_depth = np.linalg.inv(_sdata.intrinsic_depth)
        XYZ = np.empty((x.size, 3))
        XYZ[:, 0] = x_depth.flatten()
        XYZ[:, 1] = y_depth.flatten()
        XYZ[:, 2] = 1

        XYZ = inv_int_depth[:3,:3] @ XYZ.T
        XYZ[0] *= depth_img.flatten()
        XYZ[1] *= depth_img.flatten()
        XYZ[2] *= depth_img.flatten()
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(XYZ.T)
        pred_pcd.colors = o3d.utility.Vector3dVector(color_img.reshape(-1, 3)/255)
        o3d.visualization.draw_geometries([pred_pcd])


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
        x,y= np.mgrid[0:color_img.shape[0], 0:color_img.shape[1]]
        depth_img = _image.get_depth_image()
        x = np.clip(np.round(x*depth_img.shape[0]/color_img.shape[0]),0,depth_img.shape[0]-1).astype(np.int)
        y = np.clip(np.round(y*depth_img.shape[1]/color_img.shape[1]),0,depth_img.shape[1]-1).astype(np.int)
        depth_img = depth_img[x, y]
        Image.fromarray(depth_img).show()


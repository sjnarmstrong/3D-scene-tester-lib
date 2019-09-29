from segtester.util.sensreader import SensorData
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    from PIL import Image
    from time import sleep
    _sdata = SensorData("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00.sens")

    cam_params = o3d.camera.PinholeCameraIntrinsic(_sdata.depth_width, _sdata.depth_height,
                                                   _sdata.intrinsic_depth[0, 0], _sdata.intrinsic_depth[1, 1],
                                                   _sdata.intrinsic_depth[0, 2], _sdata.intrinsic_depth[1, 2])

    volume = o3d.integration.ScalableTSDFVolume(
        # voxel_length=0.048,
        voxel_length=0.0048,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    # volume = o3d.integration.UniformTSDFVolume(
    #     length=0.048,
    #     resolution=1,
    #     sdf_trunc=0.04,
    #     color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    print(_sdata)
    for i, _image in enumerate(_sdata.get_image_generator()):
        if i % 10 !=0:
            continue
        color_img = _image.get_color_image()
        depth_img = _image.get_depth_image()

        color_img = np.array(Image.fromarray(color_img).resize((_sdata.depth_width, _sdata.depth_height), Image.BICUBIC),
                           dtype=np.uint8)


        rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(o3d.geometry.Image(color_img),
                                                                          o3d.geometry.Image(depth_img),
                                               depth_scale=_sdata.depth_shift,
                                               depth_trunc=1e6, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd_image,
            cam_params,
            np.linalg.inv(_image.camera_to_world))

    occ_0 = volume.extract_voxel_point_cloud(1, -float('inf'))
    occ_1 = volume.extract_voxel_point_cloud(float('inf'), -1)
    o3d.visualization.draw_geometries([occ_0])
    o3d.visualization.draw_geometries([occ_1])

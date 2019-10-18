from segtester.util.nyuv2.raw.extract import RawDatasetArchive
from segtester.util.nyuv2.overlay import color_depth_overlay
import cv2
import numpy as np
from PIL import Image

#distortion doesnt seem worth
cmtx_depth = np.array([[582.62448168,   0.        , 313.04475871],
                        [  0.        , 582.69103271, 238.44389627],
                        [  0.        ,   0.        ,   1.        ]])
cmtx_rgb = np.array([[518.85790117,   0.        , 325.58244941],
       [  0.        , 519.46961112, 253.73616633],
       [  0.        ,   0.        ,   1.        ]])

dist_coefs_d = np.array([-0.09989724,  0.39065325,  0.00192906, -0.0019422 , -0.51031727])
dist_coefs_rgb = np.array([ 0.20796615, -0.58613825,  0.00072231,  0.00104796,  0.49856988])
R=np.array([[ 0.99997799,  0.00505184,  0.00430112],
       [-0.00503599,  0.99998052, -0.00368798],
       [-0.00431966,  0.00366624,  0.99998395]])
T=np.array([ 0.02503188,  0.00066239, -0.00029342])

cam_new_d, roi_new_d = cv2.getOptimalNewCameraMatrix(cmtx_depth, dist_coefs_d, (640, 480), 1)
cam_new_rgb, roi_new_rgb = cv2.getOptimalNewCameraMatrix(cmtx_rgb, dist_coefs_rgb, (640, 480), 1)

map_rgb_1, map_rgb_2 = cv2.initUndistortRectifyMap(cmtx_rgb, dist_coefs_rgb, R, cam_new_rgb, (640, 480), cv2.CV_32FC1)  # project rather to depth size
map_d_1, map_d_2 = cv2.initUndistortRectifyMap(cmtx_depth, dist_coefs_d, None, cam_new_d, (640, 480), cv2.CV_32FC1)

hold = RawDatasetArchive("/media/sholto/Datasets/NYUv2/nyu_depth_v2_raw.zip")
for scene_name in hold.scene_frames.keys():
    scene = hold[scene_name]
    for frame_ind in range(len(scene)):
        np_color_img = scene.load_color_image_np(frame_ind)
        np_depth_img = scene.load_depth_image_np(frame_ind)
        np_color_img_fixed = cv2.remap(np_color_img, map_rgb_1, map_rgb_2, cv2.INTER_CUBIC)
        # np_depth_img_fixed = cv2.remap(np_color_img, map_rgb_1, map_rgb_2, cv2.INTER_NEAREST)
        np_depth_img_fixed = cv2.remap(np_depth_img.astype(np.uint16), map_d_1, map_d_2, cv2.INTER_NEAREST)

        overlayed = color_depth_overlay(Image.fromarray(np_color_img_fixed, mode='RGB'), Image.fromarray(np_depth_img_fixed.astype(np.uint32), mode='I'), relative=True)
        overlayed.show()

from segtester.util.nyuv2.raw.extract import RawDatasetArchive

hold = RawDatasetArchive("/media/sholto/Datasets/NYUv2/nyu_depth_v2_raw.zip")
for scene in hold:
    for frame_ind in range(len(scene)):
        scene.load_depth_image(frame_ind).show()
        scene.load_color_image(frame_ind).show()

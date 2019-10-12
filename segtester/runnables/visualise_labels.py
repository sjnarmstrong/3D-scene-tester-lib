from typing import TYPE_CHECKING
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d
from segtester import logger
from segtester.types.dataset import Dataset
import numpy as np
import sys
import os

if TYPE_CHECKING:
    from segtester.configs.runnable.visualisepredictions import VisualisePredictionsConfig, ResultsConfig


def custom_draw_geometry_with_camera_trajectory(pcd, output_path, colors, text_labels):
    custom_draw_geometry_with_camera_trajectory.index = -1
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 935.307, 935.307, 959.5, 539.5)

    # min_pcd = np.min(pcd.points, axis=1)
    # max_pcd = np.max(pcd.points, axis=1)
    # w, h, d = max_pcd-min_pcd
    # closest_pt = min_pcd[2]

    # extrinsic_poses = [np.eye(4)]
    # trajectory = o3d.camera.PinholeCameraTrajectory()
    # trajectory_parameters = []
    # for ext_pose in extrinsic_poses:
    #     param = o3d.camera.PinholeCameraParameters()
    #     param.extrinsic = ext_pose
    #     param.intrinsic = intrinsic
    #     trajectory_parameters.append(param)
    # trajectory.parameters = trajectory_parameters
    #
    # custom_draw_geometry_with_camera_trajectory.trajectory = trajectory
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    os.makedirs(output_path, exist_ok=True)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)

            img_np = np.asarray(image)

            mask = (img_np == 1).sum(axis=2) == 3
            coords = np.array(np.nonzero(~mask))
            top_left = np.min(coords, axis=1)
            bottom_right = np.max(coords, axis=1)

            img_np = img_np[top_left[0]:bottom_right[0],
                     top_left[1]:bottom_right[1]]

            fig, ax = plt.subplots(figsize=(15, int(15 * img_np.shape[0] / img_np.shape[1]) + 2), dpi=80)
            for color, label in zip(colors, text_labels):
                ax.plot(0, 0, "-", c=color, label=label)

            ax.imshow(img_np, interpolation='nearest')
            plt.axis('off')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2),
                      ncol=4, fancybox=True, shadow=True)
            plt.savefig(f"{output_path}/labelled_view.png", bbox_inches='tight')

            plt.imsave(f"{output_path}/depth.png", np.asarray(depth), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        if glb.index < 2:
            glb.index = glb.index + 1
        if glb.index == 0:
            #ctr.scale(0.0001)
            curr_parameters = ctr.convert_to_pinhole_camera_parameters()
            hold_ext = np.array(curr_parameters.extrinsic)
            hold_ext[2,3] *= 0.82
            curr_parameters.extrinsic = hold_ext
            ctr.convert_from_pinhole_camera_parameters(curr_parameters)
        else:
            print(ctr.convert_to_pinhole_camera_parameters().extrinsic())
            # custom_draw_geometry_with_camera_trajectory.vis.\
            #        register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=1920, height=1080)
    vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json("../../TestData/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


class VisualisePredictions:
    def __init__(self, conf: 'VisualisePredictionsConfig'):
        self.cmap = plt.get_cmap("hsv")
        self.conf = conf
        self.label_map = self.conf.label_map.get_label_map()

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()

        all_class_ids = np.array(self.label_map.get_unique_values(self.conf.label_map_dest_col), dtype=np.int)
        max_to_label = all_class_ids.max()
        for scene in tqdm(dataset.scenes, desc="scene"):
            try:
                d_id = dataset_conf.dataset_id if hasattr(dataset_conf, "dataset_id") else dataset_conf.id

                save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                    "dataset_id": d_id,
                    "scene_id": scene.id, "alg_name": scene.alg_name,
                })
                print(save_path)

                if self.conf.skip_existing and os.path.exists(f"{save_path}"):
                    logger.warn(f"When getting results for 3d segmentation of {d_id}->"
                                f"{scene.id}->{scene.alg_name}, "
                                f"found existing path {save_path}.\n Skipping this scene...")
                    continue

                seg_3d_est = scene.get_seg_3d(self.label_map)

                est_label_map = self.label_map.get_label_map(scene.label_map_id_col,
                                                             self.conf.label_map_dest_col)
                clses = seg_3d_est.classes.copy()
                # clses[seg_3d_est.confidence_scores < 0.5] = 0
                seg_3d_est_labels = est_label_map[clses]
                # label_names = dataset.label_names[converted_labels]
                label_names = self.label_map.get_label_text(self.conf.label_map_dest_col, self.conf.label_map_name_col)
                all_colors = self.cmap((np.arange(max_to_label) - 2) / (max_to_label - 2))
                all_colors[0] = [0.35, 0.35, 0.35, 1]
                all_colors[0] = [0.6, 0.6, 0.6, 1]
                unique_labels = np.unique(seg_3d_est_labels)
                unique_colors = all_colors[unique_labels]
                unique_label_names = label_names[unique_labels]

                colors = self.cmap((seg_3d_est_labels-2)/(max_to_label-2))[:, :3]
                colors[seg_3d_est_labels == 0] = [0.35, 0.35, 0.35]
                colors[seg_3d_est_labels == 1] = [0.6, 0.6, 0.6]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(seg_3d_est.points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                print(f"{scene.alg_name} -- {scene.id}")
                # o3d.visualization.draw_geometries([pcd])
                custom_draw_geometry_with_camera_trajectory(pcd, save_path, unique_colors, unique_label_names)
                # o3d.visualization.draw_geometries_with_custom_animation([pcd], custom_draw_geometry_with_camera_trajectory)
                break


            except KeyboardInterrupt as e:
                try:
                    logger.error(f"Detected [ctrl+c]. Performing cleanup and then exiting...")
                    #shutil.rmtree(save_path)
                except Exception:
                    pass
                sys.exit(0)
            except Exception as e:
                try:
                    curr_gt_scene_id = None  # reload gt scene incase there is an error
                    logger.error(f"Exception when performing 3d seg assessment on {est_dataset_conf.id}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                    logger.error(e)
                    #shutil.rmtree(save_path)
                except Exception:
                    pass

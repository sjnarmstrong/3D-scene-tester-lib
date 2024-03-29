from segtester.types import Dataset
from SemanticFusion.cnn_interface.pyCaffeInterface import CaffeInterface
from SemanticFusion.semantic_fusion.pySemanticFusionInterface import SemanticFusionInterface
from SemanticFusion.map_interface.pyElasticFusionInterface import Resolution, Intrinsics, ElasticFusionInterface
from SemanticFusion.utilities.pyTypes import ClassColour, VectorOfClassColour
from SemanticFusion.gui.pyGui import Gui
from segtester import logger
from segtester.util.align_images import align_images

from tqdm import tqdm
import os
import numpy as np
import open3d as o3d
import shutil
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.alg.SemanticFusionConfig import ExecuteSemanticFusionConfig


def load_colour_scheme(filename):
    with open(filename) as fp:
        data = fp.read().split('\n')[2:]

    class_names, class_ids, rs, gs, bs, = zip(*(vals.split()[:5] for vals in data[:-1]))
    class_ids = np.array(class_ids, dtype=np.uint16)
    number_of_classes = class_ids.max()+1
    retcolormap = VectorOfClassColour(int(number_of_classes))
    for name, cls_id, r, g, b in zip(class_names, class_ids, rs, gs, bs):
        retcolormap[int(cls_id)] = ClassColour(name, int(r, 10), int(g, 10), int(b, 10))

    return retcolormap


def get_map(class_colour_lookup, save_path, conf):
    emap = ElasticFusionInterface()

    assert emap.Init(class_colour_lookup, conf.timeDelta, conf.countThresh, conf.errThresh, conf.covThresh,
                     conf.closeLoops, conf.iclnuim, conf.reloc, conf.photoThresh, conf.confidence, conf.depthCut,
                     conf.icpThresh, conf.fastOdom, conf.fernThresh, conf.so3, conf.frameToFrameRGB, save_path), \
        "Could not init emap"
    # assert emap.Init(class_colour_lookup, 200, 35000, 5e-05, 1e-05, True, True, False, 115.0, 10.0, 8.0, 10.0,
    #                  False, 0.3095, True, False, save_path), "Could not init emap"
    # assert emap.Init(class_colour_lookup, 200, 35000, 5e-05, 1e-05, True, True), "Could not init emap"
    return emap


class ExecuteSemanticFusion:
    def __init__(self, conf: 'ExecuteSemanticFusionConfig'):
        self.conf = conf
        self.caffe = CaffeInterface()
        self.caffe.Init(conf.prototext_model_path, conf.caffemodel_path, conf.caffe_use_cpu)
        self.num_classes = self.caffe.num_output_classes()
        self.semantic_fusion = SemanticFusionInterface(self.num_classes, 100)
        self.class_colour_lookup = load_colour_scheme(conf.class_colour_lookup_path)

        self.gui = Gui(True, self.class_colour_lookup, 640, 480)  # TODO not sure if that needs to be same w and h

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()
        for scene in tqdm(dataset.scenes, desc="scene"):
            try:
                logger.info(f"Processing {self.conf.alg_name} on scene {scene.id}...")

                save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                    "dataset_id": dataset_conf.id, "scene_id": scene.id,
                    "alg_name": self.conf.alg_name
                })

                if self.conf.skip_existing and os.path.exists(f"{save_path}"):
                    logger.warn(f"When processing "
                                f"{dataset_conf.id}->{scene.id}->{self.conf.alg_name}, "
                                f"found existing path {save_path}.\n Skipping this scene...")
                    continue
                # intrinsic_res = scene.get_intrinsic_rgb()
                # height, width = scene.get_rgb_size()
                intrinsic_res = scene.get_intrinsic_depth()
                height, width = scene.get_depth_size()
                Resolution.getInstance(width, height)
                Intrinsics.getInstance(intrinsic_res[0, 0], intrinsic_res[1, 1],
                                       intrinsic_res[0, 2], intrinsic_res[1, 2])

                emap = get_map(self.class_colour_lookup, f"{save_path}/elastic_generated", self.conf)
                frame_save_path = f"{save_path}/frames"
                os.makedirs(frame_save_path, exist_ok=True)
                pose_array = np.empty((scene.get_num_frames(), 16), dtype=np.float32)

                init_world_to_cam = None
                init_cam_to_world = None
                for rgb, depth, camera_to_world, timestamp, frame_num in \
                        tqdm(scene.get_rgb_depth_image_it(), desc="frame", total=scene.get_num_frames()):
                    # if frame_num == 200:
                    #     break
                    if frame_num == 0:
                        init_cam_to_world = camera_to_world
                        init_world_to_cam = np.linalg.inv(camera_to_world)

                    self.gui.preCall()
                    rgb = align_images(rgb, width, height)
                    rgb, depth = rgb.flatten(), depth.flatten()

                    # Semantic fusion expects a depth scale of 1000
                    if scene.get_depth_scale() != 1000:
                       depth = np.round(depth.astype(np.float) * 1000 / scene.get_depth_scale()).astype(np.uint16)
                    # depth = 1092.5 - (351.3 * scene.get_depth_scale() / depth.astype(np.float))
                    # depth[np.isnan(depth)] = 65535
                    # depth = np.round(depth).astype(np.uint16)

                    # convert timestamp format from python to one of NYu
                    if self.conf.use_gt_pose:
                        if not emap.ProcessFrameNumpy(rgb, depth, int(timestamp*1000), init_world_to_cam @ camera_to_world):
                            raise Exception("Elastic fusion lost!")
                    else:
                        if not emap.ProcessFrameNumpy(rgb, depth, int(timestamp*1000), np.empty([0,0], dtype=np.float32)):
                            raise Exception("Elastic fusion lost!")

                    self.semantic_fusion.UpdateProbabilityTable(emap)
                    if frame_num == 0 or (frame_num > 1 and ((frame_num + 1) % self.conf.cnn_skip_frames == 0)):
                        predicted_probs = \
                            self.semantic_fusion.PredictAndUpdateProbabilities(rgb, depth,
                                                                               self.caffe, emap, True)
                    if self.conf.save_frames:
                        np.savez_compressed(f"{frame_save_path}/frame_{frame_num}",
                                            likelihoods=predicted_probs[:, :, :, 0])
                    if self.conf.use_crf and frame_num % self.conf.crf_skip_frames == 0:
                        print("Performing CRF Update...")
                        self.semantic_fusion.CRFUpdate(emap, self.conf.crf_iterations)

                    pose_array[frame_num] = emap.getCurrentPose().flat

                    self.gui.renderMap(emap)
                    self.semantic_fusion.CalculateProjectedProbabilityMap(emap)  # note I think this may be important
                    self.gui.displayArgMaxClassColouring(
                        "segmentation", self.num_classes, 0.0, emap, self.semantic_fusion)
                    self.gui.displayImg("raw", emap)

                    self.gui.postCall()

                print("here")
                xyz, rgb, pr = self.semantic_fusion.GetGlobalMap(emap)
                pred_pcd = o3d.geometry.PointCloud()
                xyz_hold = xyz[:, :4].copy()
                xyz_hold[:, 3] = 1
                xyz_hold = xyz_hold @ init_cam_to_world.T
                pred_pcd.points = o3d.utility.Vector3dVector(xyz_hold[:, :3])
                pred_pcd.colors = o3d.utility.Vector3dVector(rgb/255)
                pred_pcd.normals = o3d.utility.Vector3dVector(xyz[:, 3:])
                o3d.io.write_point_cloud(f"{save_path}/pcd.ply", pred_pcd)
                np.savez_compressed(f"{save_path}/probs", likelihoods=pr)
                np.savez_compressed(f"{save_path}/poses", pose_array=pose_array, init_cam_to_world=init_cam_to_world)

            except KeyboardInterrupt as e:
                logger.error(f"Detected [ctrl+c]. Performing cleanup and then exiting...")
                try:
                    shutil.rmtree(save_path)
                except Exception:
                    pass
                sys.exit(0)
            except Exception as e:
                logger.exception(f"Exception when running {self.conf.alg_name} on {dataset_conf.id}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                logger.exception(str(e))
                try:
                    shutil.rmtree(save_path)
                except Exception:
                    pass

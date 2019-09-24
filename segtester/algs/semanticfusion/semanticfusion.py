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


def get_map(class_colour_lookup):
    emap = ElasticFusionInterface()
    assert emap.Init(class_colour_lookup), "Could not init emap"
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
                # intrinsic_res = scene.get_intrinsic_rgb()
                # height, width = scene.get_rgb_size()
                intrinsic_res = scene.get_intrinsic_depth()
                height, width = scene.get_depth_size()
                Resolution.getInstance(width, height)
                Intrinsics.getInstance(intrinsic_res[0, 0], intrinsic_res[1, 1],
                                       intrinsic_res[0, 2], intrinsic_res[0, 1])

                emap = get_map(self.class_colour_lookup)

                save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                    "dataset_id": dataset_conf.id, "scene_id": scene.id,
                    "alg_name": self.conf.alg_name
                })
                frame_save_path = f"{save_path}/frames"
                os.makedirs(frame_save_path, exist_ok=True)
                for rgb, depth, timestamp, frame_num in \
                        tqdm(scene.get_rgb_depth_image_it(), desc="frame", total=scene.get_num_frames()):

                    self.gui.preCall()
                    rgb = align_images(rgb, width, height)
                    rgb, depth = rgb.flatten(), depth.flatten()

                    # convert timestamp format from python to one of NYu
                    if not emap.ProcessFrameNumpy(rgb, depth, int(timestamp*1000)):
                        raise Exception("Elastic fusion lost!")

                    self.semantic_fusion.UpdateProbabilityTable(emap)
                    if frame_num == 0 or (frame_num > 1 and ((frame_num + 1) % self.conf.cnn_skip_frames == 0)):
                        predicted_probs = \
                            self.semantic_fusion.PredictAndUpdateProbabilities(rgb, depth,
                                                                               self.caffe, emap, True)
                    np.savez_compressed(f"{frame_save_path}/frame_{frame_num}", likelihoods=predicted_probs[:, :, :, 0])
                    if self.conf.use_crf and frame_num % self.conf.crf_skip_frames == 0:
                        print("Performing CRF Update...")
                        self.semantic_fusion.CRFUpdate(emap, self.conf.crf_iterations)

                    self.gui.renderMap(emap)
                    self.semantic_fusion.CalculateProjectedProbabilityMap(emap)  # note I think this may be important
                    self.gui.displayArgMaxClassColouring(
                        "segmentation", self.num_classes, 0.0, emap, self.semantic_fusion)
                    self.gui.displayImg("raw", emap)

                    self.gui.postCall()
                    if frame_num == 100:
                        break

                xyz, rgb, pr = self.semantic_fusion.GetGlobalMap(emap)
                pred_pcd = o3d.geometry.PointCloud()
                pred_pcd.points = o3d.utility.Vector3dVector(xyz)
                pred_pcd.colors = o3d.utility.Vector3dVector(rgb/255)
                o3d.io.write_point_cloud(f"{save_path}/pcd.ply", pred_pcd)
                np.savez_compressed(f"{save_path}/probs", likelihoods=pr)

            except Exception as e:
                logger.error(f"Exception when running SemanticFusion on {dataset_conf.id}:{scene.id}. "
                             f"Skipping scene and moving on...")
                logger.error(str(e))

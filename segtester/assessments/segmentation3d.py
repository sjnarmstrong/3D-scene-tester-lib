from segtester.types import Scene, Dataset
from tqdm import tqdm
from segtester import logger
import shutil
import numpy as np
import os
import sys
import segtester.metrics.seg as smet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.assessments.seg3d import Segmentation3d as Config


class SubTestResult:
    def __init__(self, **kwargs):
        self.pt_acc_num = kwargs["pt_acc_num"]
        self.pt_acc_den = kwargs["pt_acc_den"]
        self.inst_acc_num = kwargs["inst_acc_num"]
        self.inst_acc_den = kwargs["inst_acc_den"]
        self.mca = kwargs["mca"]
        self.iou = kwargs["iou"]
        self.miou = kwargs["miou"]
        self.fiou = kwargs["fiou"]


class TestResult:
    def __init__(self, **kwargs):
        self.seg_name: str = kwargs["seg_name"]
        self.res_class: SubTestResult = kwargs["res_class"]
        self.res_inst: SubTestResult = kwargs["res_inst"]
        self.res_seg: SubTestResult = kwargs["res_seg"]
        self.inst_precision: np.array = kwargs["inst_precision"]
        self.inst_recall: np.array = kwargs["inst_recall"]
        self.precision_recall_probs: list = kwargs["precision_recall_probs"]
        self.precision_recall_iou: np.array = kwargs["precision_recall_iou"]


class Segmentation3DAssessment:
    def __init__(self, conf: 'Config'):
        self.conf = conf
        self.label_map = self.conf.label_map.get_label_map()

    def __call__(self, base_result_path, gt_dataset_conf, est_dataset_conf, *_, **__):
        gt_dataset: Dataset = gt_dataset_conf.get_dataset()
        est_dataset: Dataset = est_dataset_conf.get_dataset()
        scenes_by_id = {}
        for scene in est_dataset.scenes:
            if scene.id not in scenes_by_id:
                scenes_by_id[scene.id] = []
            scenes_by_id[scene.id].append(scene)
        scenes_sorted_by_id = (scene for scene_id in scenes_by_id.keys() for scene in scenes_by_id[scene_id])

        all_class_ids = np.array(self.label_map.get_unique_values(self.conf.label_map_dest_col), dtype=np.int)
        curr_gt_scene_id = None
        seg_3d_gt = None
        gt_label_map = None
        for scene in tqdm(scenes_sorted_by_id, desc="scene"):
            try:

                save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                    "dataset_id": est_dataset_conf.id, "scene_id": scene.id,
                    "alg_name": scene.alg_name,
                })

                if self.conf.skip_existing and os.path.exists(f"{save_path}"):
                    logger.warn(f"When getting results for 3d segmentation of {est_dataset_conf.id}->"
                                f"{scene.id}->{scene.alg_name}, "
                                f"found existing path {save_path}.\n Skipping this scene...")
                    continue

                if curr_gt_scene_id != scene.id:
                    curr_gt_scene_id = scene.id
                    for curr_gt_scene in gt_dataset.scenes:
                        if curr_gt_scene.id == curr_gt_scene_id:
                            break
                    else:
                        curr_gt_scene_id = None
                        raise Exception(f"Could not find gt scene for scene id {curr_gt_scene_id}")
                    seg_3d_gt = curr_gt_scene.get_seg_3d(self.label_map)
                    gt_label_map = self.label_map.get_label_map(curr_gt_scene.label_map_id_col,
                                                                self.conf.label_map_dest_col)

                seg_3d_est = scene.get_seg_3d(self.label_map)

                est_label_map = self.label_map.get_label_map(scene.label_map_id_col,
                                                             self.conf.label_map_dest_col)
                seg_3d_est_labels = est_label_map[seg_3d_est.classes]

                mapped_gt_seg = seg_3d_gt.get_mapped_seg(gt_label_map, seg_3d_est)

                res_class = Segmentation3DAssessment.get_results(seg_3d_est_labels,
                                                                 mapped_gt_seg.classes, all_class_ids)
                est_labels, gt_labels, _ = seg_3d_est.get_segmentation_labels(mapped_gt_seg, match_classes=True)
                res_inst = Segmentation3DAssessment.get_results(est_labels, gt_labels)

                precision, recall, test_probs, iou_threshs = \
                    smet.precision_recall(est_labels, gt_labels, seg_3d_est.confidence_scores)

                est_labels, gt_labels, _ = seg_3d_est.get_segmentation_labels(mapped_gt_seg, match_classes=False)
                res_seg = Segmentation3DAssessment.get_results(est_labels, gt_labels)

                np.savez_compressed(f"{save_path}/res_class", **res_class)
                np.savez_compressed(f"{save_path}/res_class", **res_inst)
                np.savez_compressed(f"{save_path}/res_class", **res_seg)
                np.savez_compressed(f"{save_path}/pvr", precision=precision, recall=recall,
                                    test_probs=test_probs, iou_threshs=iou_threshs)

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

    @staticmethod
    def get_results(est_labels, gt_labels, class_ids=None):
        pt_acc_num, pt_acc_den = smet.point_accuracy(est_labels, gt_labels)
        inst_acc_num, inst_acc_den, _ = smet.class_accuracy(est_labels, gt_labels, class_ids)
        mca = smet.mean_class_accuracy(inst_acc_num / inst_acc_den)
        iou = smet.iou(est_labels, gt_labels, class_ids).tolist()
        miou = smet.miou(iou)
        fiou = smet.fiou(est_labels, gt_labels, class_ids)
        res = {
            "pt_acc_num": pt_acc_num, "pt_acc_den": pt_acc_den, "inst_acc_num": inst_acc_num,
            "inst_acc_den": inst_acc_den, "mca": mca, "iou": iou, "miou": miou, "fiou": fiou
        }
        return res
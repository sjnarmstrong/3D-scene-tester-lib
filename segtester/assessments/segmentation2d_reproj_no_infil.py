from segtester.types import Scene, Dataset
from tqdm import tqdm
from segtester import logger
import shutil
import numpy as np
import os
import sys
import segtester.metrics.seg as smet
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image
from segtester.types.seg2d import Seg2D

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


class Segmentation2DReprojAssessment:
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

        all_class_ids = np.array(self.label_map.get_unique_values(self.conf.label_map_dest_col), dtype=np.int)
        for scene_id in tqdm(scenes_by_id.keys(), desc="scene_id"):
            try:
                gt_scene = next(it for it in gt_dataset.scenes if it.id == scene_id)
                gt_label_map = self.label_map.get_label_map(gt_scene.label_map_id_col,
                                                            self.conf.label_map_dest_col)

                cashed_est_3dseg = []
                for scene_it in scenes_by_id[scene_id]:
                    seg_3d_est = scene_it.get_seg_3d(self.label_map)

                    est_label_map = self.label_map.get_label_map(scene_it.label_map_id_col,
                                                                 self.conf.label_map_dest_col)
                    seg_3d_est.map_own_classes(est_label_map)
                    cashed_est_3dseg.append(seg_3d_est)

                for reproj_gt_seg, s2d_gt, seg_inds, img_nr in gt_scene.get_labelled_reproj_seg3d(est_scenes=scenes_by_id[scene_id], est_seg3d=seg_3d_est):
                    reproj_gt_seg.map_own_classes(gt_label_map)
                    s2d_gt.map_own_classes(gt_label_map)

                    for scene, seg_3d_est in zip(scenes_by_id[scene_id], cashed_est_3dseg):

                        save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                            "dataset_id": gt_dataset_conf.id, "scene_id": scene.id,
                            "alg_name": scene.alg_name,
                        })

                        if self.conf.skip_existing and os.path.exists(f"{save_path}/pvr_{img_nr:06d}.npz"):
                            logger.warn(f"When getting results for 3d segmentation of {est_dataset_conf.id}->"
                                        f"{scene.id}->{scene.alg_name}, "
                                        f"found existing path {save_path}.\n Skipping this scene...")
                            continue

                        mapped_est_seg, dists = seg_3d_est.get_mapped_seg(reproj_gt_seg)
                        dist_mask = dists.flatten() < self.conf.point_dist_thresh
                        seg_inds = seg_inds[:, dist_mask]
                        mapped_labels = np.zeros(s2d_gt.image_shape, dtype=np.int)
                        mapped_labels[seg_inds[0], seg_inds[1]] = mapped_est_seg.classes[dist_mask]
                        mapped_segs = np.zeros((mapped_est_seg.instance_masks.shape[0],) + s2d_gt.image_shape,
                                               dtype=np.bool)
                        mapped_segs[:, seg_inds[0], seg_inds[1]] = mapped_est_seg.instance_masks[:, dist_mask]
                        seg_mask = np.any(mapped_segs, axis=(1, 2))

                        s2d_est = Seg2D(
                            mapped_labels, mapped_segs[seg_mask], mapped_est_seg.instance_classes[seg_mask],
                            np.ones(mapped_labels.size)
                        )

                        res_class = Segmentation2DReprojAssessment.get_results(s2d_est.classes, s2d_gt.classes,
                                                                               all_class_ids)
                        est_labels, gt_labels, _ = s2d_est.get_segmentation_labels(s2d_gt, match_classes=True)
                        res_inst = Segmentation2DReprojAssessment.get_results(est_labels, gt_labels)

                        precision, recall, test_probs, iou_threshs = \
                            smet.precision_recall(est_labels, gt_labels, s2d_est.confidence_scores)

                        est_labels, gt_labels, _ = s2d_est.get_segmentation_labels(s2d_gt, match_classes=False)
                        res_seg = Segmentation2DReprojAssessment.get_results(est_labels, gt_labels)

                        os.makedirs(save_path, exist_ok=True)
                        print(f"{save_path}/res_class_{img_nr:06d}")
                        s2d_gt.vis_labels().save(f"{save_path}/gt_{img_nr:06d}.png")
                        s2d_est.vis_labels().save(f"{save_path}/est_{img_nr:06d}.png")

                        np.savez_compressed(f"{save_path}/res_class_{img_nr:06d}", **res_class)
                        np.savez_compressed(f"{save_path}/res_inst_{img_nr:06d}", **res_inst)
                        np.savez_compressed(f"{save_path}/res_seg_{img_nr:06d}", **res_seg)
                        np.savez_compressed(f"{save_path}/pvr_{img_nr:06d}", precision=precision, recall=recall,
                            test_probs=test_probs, iou_threshs=iou_threshs)

                        # # visualise reprojected labels
                        # cmap = plt.get_cmap("hsv")
                        # pcd = o3d.geometry.PointCloud()
                        # pcd.points = o3d.utility.Vector3dVector(reproj_gt_seg.points)
                        # colors = np.ones_like(reproj_gt_seg.points)
                        # for i, inst_mask in enumerate(reproj_gt_seg.instance_masks):
                        #     colors[inst_mask] = cmap(i/len(reproj_gt_seg.instance_masks))[:3]
                        # pcd.colors = o3d.utility.Vector3dVector(colors)
                        # pcd2 = gt_scene.get_pcd()
                        # o3d.visualization.draw_geometries([pcd,pcd2])
                        # max_class = max(mapped_labels.max(), s2d_gt.classes.max())
                        # vis_cmap = cmap(np.arange(max_class+1)/(max_class+1))
                        # vis_cmap[0] = (0.3, 0.3, 0.3, 1)
                        # Image.fromarray((vis_cmap[mapped_labels][:, :, :3] * 255).astype(np.uint8)).show()
                        # Image.fromarray((vis_cmap[s2d_gt.classes].reshape(s2d_gt.image_shape+(4,))[:, :, :3] * 255).astype(np.uint8)).show()


            except Exception as e:
                curr_gt_scene_id = None  # reload gt scene incase there is an error
                logger.error(f"Exception when performing 3d seg assessment on {est_dataset_conf.id}:{scene.id}. "
                             f"Skipping scene and moving on...")
                logger.error(e)

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
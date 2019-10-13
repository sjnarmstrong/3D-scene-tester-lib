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
    from segtester.configs.assessments.summarizeres import SummarizeRes as Config


class ReconstructionSummaryResults:
    def __init__(self, alg_names, scene_ids):
        self.all_error_means = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))
        self.all_error_stds = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))
        self.all_error_mse = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))

    def add_result(self, base_path, alg_ind, scene_ind):
        path = f"{base_path}/seg/seg3d/point_dists.npz"
        print(path)
        try:
            if not os.path.exists(path):
                return
            file = np.load(path)
            point_dists = file["point_dists"]
            self.all_error_means[alg_ind][scene_ind] = np.nanmean(point_dists)
            self.all_error_stds[alg_ind][scene_ind] = np.nanstd(point_dists)
            self.all_error_mse[alg_ind][scene_ind] = np.nanmean(point_dists**2)

        except Exception as e:
            pass


class ReconstructionSeg3dBasicResults:
    def __init__(self, alg_names, scene_ids, t=0):
        if t == 0:
            self.file_to_use = "res_class.npz"
        elif t == 1:
            self.file_to_use = "res_inst.npz"
        else:
            self.file_to_use = "res_seg.npz"
        self.pt_accuracy = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))
        self.inst_accuracy = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))
        self.mca = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))
        self.fiou = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))
        self.miou = np.repeat(np.nan, len(alg_names)*len(scene_ids)).reshape(len(alg_names), len(scene_ids))

    def add_result(self, base_path, alg_ind, scene_ind):
        path = f"{base_path}/seg/seg3d/{self.file_to_use}"
        try:
            if not os.path.exists(path):
                return
            file = np.load(path)
            self.pt_accuracy[alg_ind][scene_ind] = file["pt_acc_num"]/file["pt_acc_den"]
            self.inst_accuracy[alg_ind][scene_ind] = file["inst_acc_num"]/file["inst_acc_den"]
            self.mca[alg_ind][scene_ind] = file["mca"]
            self.fiou[alg_ind][scene_ind] = file["fiou"]
            self.miou[alg_ind][scene_ind] = file["miou"]
        except Exception as e:
            pass


class SummarizeRes:
    def __init__(self, conf: 'Config'):
        self.conf = conf
        self.label_map = self.conf.label_map.get_label_map()

    def __call__(self, base_result_path, est_dataset_conf, *_, **__):
        dataset = est_dataset_conf.get_dataset()
        rec_res = ReconstructionSummaryResults(dataset.alg_names, dataset.scene_ids)
        alg_name_map = {k:v for (v,k) in enumerate(dataset.alg_names)}
        scene_id_map = {k:v for (v,k) in enumerate(dataset.scene_ids)}

        for scene in tqdm(dataset.scenes, desc="scene"):
            try:
                base_path = scene.base_path
                scene_ind = scene_id_map[scene.id]
                alg_ind = alg_name_map[scene.alg_name]
                rec_res.add_result(base_path, alg_ind, scene_ind)

                # if os.path.exists(f"{base_path}/seg/seg3d/point_dists"):


                    # np.savez_compressed(f"{save_path}/point_dists", point_dists=point_dists)
                    # np.savez_compressed(f"{save_path}/res_class", **res_class)
                    # np.savez_compressed(f"{save_path}/res_inst", **res_inst)
                    # np.savez_compressed(f"{save_path}/res_seg", **res_seg)
                    # np.savez_compressed(f"{save_path}/pvr", precision=precision, recall=recall,
                    #                     test_probs=test_probs, iou_threshs=iou_threshs)

            except Exception as e:
                logger.error(f"Exception when performing 3d seg assessment on {est_dataset_conf.id}:{scene.id}. "
                             f"Skipping scene and moving on...")
                logger.error(e)

from segtester.types import Scene, Dataset
from tqdm import tqdm
from segtester import logger
import shutil
import numpy as np
import os
import sys
import segtester.metrics.seg as smet
from zipfile import ZipFile
from glob import glob
import pandas as pd
import json

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.assessments.summarizeres import SummarizeRes as Config


class Seg2DResults:
    def __init__(self, t=0):
        self.per_class_results = None
        if t == 0:
            self.file_to_use = "res_class_*.npz"
            self.per_class_results = pd.DataFrame(columns=['alg_name', 'scene', 'frame_nr', 'class', 'iou',
                                                           'inst_acc_num', 'inst_acc_den'])
        elif t == 1:
            self.file_to_use = "res_inst_*.npz"
        else:
            self.file_to_use = "res_seg_*.npz"
        self.all_results = pd.DataFrame(columns=[
            'alg_name', 'scene', 'frame_nr', 'pt_acc_num', 'mca', 'fiou', 'miou'
        ])

    def add_result(self, base_path, alg_name, scene_name, classes):
        try:
            paths = glob(f"{base_path}/seg/seg2d/{self.file_to_use}")
            for path in paths:
                file = np.load(path)
                frame_nr = int(path.split('_')[-1].split('.npz')[0])
                self.all_results = self.all_results.append({
                    'alg_name': alg_name, 'scene': scene_name, 'frame_nr': frame_nr,
                    'pt_acc_num': file["pt_acc_num"],  'mca': file["mca"], 'fiou': file["fiou"], 'miou': file["miou"]
                }, ignore_index=True)
                if self.per_class_results is not None:
                    inst_acc_num = file["inst_acc_num"]
                    inst_acc_den = file["inst_acc_den"]
                    for i, iou in enumerate(file["iou"]):
                        self.per_class_results = self.per_class_results.append({
                            'alg_name': alg_name, 'scene': scene_name, 'frame_nr': frame_nr, 'class': classes[i],
                            'iou': iou, 'inst_acc_num': inst_acc_num[i], 'inst_acc_den': inst_acc_den[i],
                        }, ignore_index=True)
        except Exception as e:
            print("2d")
            print(e)
            print(e.__traceback__)


class ReconstructionResults:
    def __init__(self):
        self.all_results = pd.DataFrame(columns=[
            'alg_name', 'scene', 'reconstruction_error_mean', 'reconstruction_error_std', 'reconstruction_error_mse',
        ])

    def add_result(self, base_path, alg_name, scene_name):
        try:
            path = f"{base_path}/seg/seg3d/point_dists.npz"
            if not os.path.exists(path):
                return
            file = np.load(path)
            point_dists = file["point_dists"]
            self.all_results = self.all_results.append({
                'alg_name': alg_name, 'scene': scene_name,
                "reconstruction_error_mean": np.nanmean(point_dists),
                "reconstruction_error_std": np.nanstd(point_dists),
                "reconstruction_error_mse": np.nanmean(point_dists ** 2)
            }, ignore_index=True)

        except Exception as e:
            print("rec")
            print(e)
            print(e.__traceback__)



class OdometryRes:
    FILES_TO_USE = [
        "noalign_full_transformationape.zip",
        "noalign_full_transformationrpe.zip",
        "noalign_rotation_partape.zip",
        "noalign_rotation_partrpe.zip",
        "noalign_translation_partape.zip",
        "noalign_translation_partrpe.zip",
    ]

    def __init__(self, t=0):
        self.file_to_use = OdometryRes.FILES_TO_USE[t]
        self.all_results = pd.DataFrame(columns=[
            "rmse", "mean", "median", "std", "min", "max", "sse",
        ])

    def add_result(self, base_path, alg_name, scene_name):
        try:
            path = f"{base_path}/odo/{self.file_to_use}"
            if not os.path.exists(path):
                return
            with ZipFile(path) as fp:
                stat_data = json.loads(fp.read("stats.json"))

            self.all_results = self.all_results.append({
                'alg_name': alg_name, 'scene': scene_name,
                "rmse": stat_data["rmse"],
                "mean": stat_data["mean"],
                "median": stat_data["median"],
                "std": stat_data["std"],
                "min": stat_data["min"],
                "max": stat_data["max"],
                "sse": stat_data["sse"],
            }, ignore_index=True)

        except Exception as e:
            print("rec")
            print(e)
            print(e.__traceback__)

class ReconstructionSeg3dBasicResults:
    def __init__(self, alg_names, scene_ids, classes, t=0):
        data_total = len(alg_names) * len(scene_ids)
        self.per_class_results = None
        if t == 0:
            self.file_to_use = "res_class.npz"
            per_class_data = {
                "alg_name": np.repeat(alg_names, len(scene_ids)*len(classes)),
                "scene": np.repeat(np.tile(scene_ids, len(alg_names)), len(classes)),
                "class": np.tile(classes, data_total),
                "iou": np.repeat(np.nan, data_total*len(classes)),
                "inst_acc_num": np.repeat(np.nan, data_total*len(classes)),
                "inst_acc_den": np.repeat(np.nan, data_total*len(classes)),
            }
            self.per_class_results = pd.DataFrame(per_class_data)
        elif t == 1:
            self.file_to_use = "res_inst.npz"
        else:
            self.file_to_use = "res_seg.npz"
        data = {"alg_name": np.repeat(alg_names, len(scene_ids)),
                "scene": np.tile(scene_ids, len(alg_names)),
                "pt_acc_num": np.repeat(np.nan, data_total),
                "mca": np.repeat(np.nan, data_total),
                "fiou": np.repeat(np.nan, data_total),
                "miou": np.repeat(np.nan, data_total),
                }
        self.all_results = pd.DataFrame(data)

    def add_result(self, base_path, alg_name, scene_name):
        path = f"{base_path}/seg/seg3d/{self.file_to_use}"
        try:
            if not os.path.exists(path):
                return
            file = np.load(path)
            pos_f = (self.all_results['alg_name'] == alg_name) & (self.all_results['scene'] == scene_name)
            self.all_results.at[pos_f, "pt_acc_num"] = file["pt_acc_num"]
            self.all_results.at[pos_f, "pt_acc_den"] = file["pt_acc_den"]
            self.all_results.at[pos_f, "mca"] = file["mca"]
            self.all_results.at[pos_f, "fiou"] = file["fiou"]
            self.all_results.at[pos_f, "miou"] = file["miou"]
            if self.per_class_results is not None:
                pos_f = (self.per_class_results['alg_name'] == alg_name) & \
                        (self.per_class_results['scene'] == scene_name)
                self.per_class_results.loc[pos_f, ["iou"]] = file["iou"]
                self.per_class_results.loc[pos_f, ["inst_acc_num"]] = file["inst_acc_num"]
                self.per_class_results.loc[pos_f, ["inst_acc_den"]] = file["inst_acc_den"]
        except Exception as e:
            print("3d")
            print(e)
            print(e.__traceback__)


class SummarizeRes:
    name_map = {'MinkUNet34C_0.02': 'MinkUNet34C (2cm)', 'MinkUNet34C_0.05': 'MinkUNet34C (5cm)',
                'SemanticFusion': 'SemanticFusion', '3DMV': '3DMV'
                }

    def __init__(self, conf: 'Config'):
        self.conf = conf
        self.label_map = self.conf.label_map.get_label_map()

    def __call__(self, base_result_path, est_dataset_conf, *_, **__):
        classes = self.label_map.get_label_text(self.conf.label_map_dest_col, self.conf.label_map_dest_name_col)
        dataset = est_dataset_conf.get_dataset()
        seg3d_res_class = ReconstructionSeg3dBasicResults(dataset.alg_names, dataset.scene_ids, classes, 0)
        seg3d_res_inst = ReconstructionSeg3dBasicResults(dataset.alg_names, dataset.scene_ids, classes, 1)
        seg3d_res_seg = ReconstructionSeg3dBasicResults(dataset.alg_names, dataset.scene_ids, classes, 2)
        seg2d_res_class = Seg2DResults(0)
        seg2d_res_inst = Seg2DResults(1)
        seg2d_res_seg = Seg2DResults(2)
        reconst_res = ReconstructionResults()
        odo_res = OdometryRes()

        for scene in tqdm(dataset.scenes, desc="scene"):
            try:
                base_path = scene.base_path
                seg3d_res_class.add_result(base_path, scene.alg_name, scene.id)
                seg3d_res_inst.add_result(base_path, scene.alg_name, scene.id)
                seg3d_res_seg.add_result(base_path, scene.alg_name, scene.id)
                seg2d_res_class.add_result(base_path, scene.alg_name, scene.id, classes)
                seg2d_res_inst.add_result(base_path, scene.alg_name, scene.id, classes)
                seg2d_res_seg.add_result(base_path, scene.alg_name, scene.id, classes)
                reconst_res.add_result(base_path, scene.alg_name, scene.id)
                odo_res.add_result(base_path, scene.alg_name, scene.id)
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

        try:
            save_folder = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                "dataset_id": self.conf.dataset_id
            })
            os.makedirs(save_folder, exist_ok=True)
            # Per class Results
            self.create_seg_per_class_tex(f"{save_folder}/seg2d_per_class_acc.tex",
                                          f"{save_folder}/seg2d_per_class_iou.tex",
                                          seg2d_res_class.per_class_results)
            self.create_seg_per_class_tex(f"{save_folder}/seg3d_per_class_acc.tex",
                                          f"{save_folder}/seg3d_per_class_iou.tex",
                                          seg3d_res_class.per_class_results)
            # All results
            self.create_seg_all_res_tex(f"{save_folder}/seg2d_all_c.tex", seg2d_res_class.all_results)
            self.create_seg_all_res_tex(f"{save_folder}/seg3d_all_c.tex", seg3d_res_class.all_results)
            self.create_seg_all_res_tex(f"{save_folder}/seg2d_all_i.tex", seg2d_res_inst.all_results)
            self.create_seg_all_res_tex(f"{save_folder}/seg3d_all_i.tex", seg3d_res_inst.all_results)
            self.create_seg_all_res_tex(f"{save_folder}/seg2d_all_s.tex", seg2d_res_seg.all_results)
            self.create_seg_all_res_tex(f"{save_folder}/seg3d_all_s.tex", seg3d_res_seg.all_results)

            # Reconstruction
            self.create_reconstruction_tex(f"{save_folder}/rec_main.tex", reconst_res.all_results)

            # ODOM
            self.create_odom_tex(f"{save_folder}/odom_main.tex", odo_res.all_results)

        except Exception as e:
            logger.error(f"Could not save tex files")
            logger.exception(e)

    @classmethod
    def create_seg_all_res_tex(cls, save_path, df):
        try:
            if len(df) == 0:
                return
            table = df.pivot_table(values=["miou", "fiou"], index=["alg_name"], aggfunc=np.nanmean)
            t2 = df.pivot_table(values=["pt_acc_num", "pt_acc_den"], index=["alg_name"], aggfunc=np.nansum)
            table.insert(0, 'Accuracy (%)', 100 * t2["pt_acc_num"] / t2["pt_acc_den"])

            table = table.rename(cls.name_map).replace(0, np.NaN)
            overall_accuracy_tex = table.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x) \
                .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm").replace("fiou", "FIoU") \
                .replace("miou", "MIoU")
            with open(save_path, "w") as fp:
                fp.write(overall_accuracy_tex)
        except Exception as e:
            logger.error(f"Could not save tex files")
            logger.exception(e)

    @classmethod
    def create_seg_per_class_tex(cls, save_path_acc, save_path_iou, df):
        try:
            if len(df) == 0:
                return
            table = df.pivot_table(values="iou", index=["alg_name"], columns=["class"], aggfunc=np.nanmean)

            table = table.rename(cls.name_map).replace(0, np.NaN)
            per_class_iou_tex = table.iloc[:, :table.shape[1] - 1].T.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x) \
                .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm")
            with open(save_path_iou, "w") as fp:
                fp.write(per_class_iou_tex)

            table = df.pivot_table(values=["inst_acc_num", "inst_acc_den"], index=["alg_name"], columns=["class"], aggfunc=np.sum)
            table = 100 * table["inst_acc_num"] / table["inst_acc_den"]
            table = table.rename(cls.name_map).replace(0, np.NaN)
            per_class_acc_tex = table.iloc[:, :table.shape[1] - 1].T.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x) \
                .replace('nan', '-').replace("class ", "Class ").replace("alg\\_name", "Algorithm")
            with open(save_path_acc, "w") as fp:
                fp.write(per_class_acc_tex)
        except Exception as e:
            logger.error(f"Could not save tex files")
            logger.exception(e)

    @classmethod
    def create_reconstruction_tex(cls, save_path, df):
        try:
            if len(df) == 0:
                return
            df["rec_var"] = df["reconstruction_error_std"] ** 2
            table = df.pivot_table(
                values=["reconstruction_error_mean", "reconstruction_error_mse", "rec_var"],
                index=["alg_name"], aggfunc=np.nanmean)
            t1 = df.pivot_table(values=["reconstruction_error_mean"], index=["alg_name"], aggfunc=np.nanmax)
            table["STD. (m)"] = np.sqrt(table["rec_var"])
            table["Max Mean Error (m)"] = t1["reconstruction_error_mean"]
            table["RMSE (m)"] = np.sqrt(table["reconstruction_error_mse"])
            table = table.rename(cls.name_map).replace(0, np.NaN)
            reconstruction_res_tex = table.to_latex(columns=["reconstruction_error_mean", "STD. (m)", "Max Mean Error (m)",
                                                             "RMSE (m)"],
                                                    na_rep='-',
                                                    float_format=lambda x: '%.3f' % x) \
                .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm") \
                .replace("reconstruction\\_error\\_mean", "Mean (m)")
            with open(save_path, "w") as fp:
                fp.write(reconstruction_res_tex)
        except Exception as e:
            logger.error(f"Could not save tex files")
            logger.exception(e)

    @classmethod
    def create_odom_tex(cls, save_path, df):
        try:
            if len(df) == 0:
                return

            df["var"] = df["std"] ** 2
            df["mse"] = df["rmse"] ** 2
            table = df.pivot_table(values=["mean", "mse", "var"], index=["alg_name"], aggfunc=np.nanmean)
            table['max'] = df.pivot_table(values=["max"], index=["alg_name"], aggfunc=np.nanmax)['max']
            table['min'] = df.pivot_table(values=["min"], index=["alg_name"], aggfunc=np.nanmin)['min']
            table['rmse'] = np.sqrt(table["mse"])
            table['std'] = np.sqrt(table["var"])
            table = table.rename(cls.name_map).replace(0, np.NaN)
            odom_res_tex = table.to_latex(columns=["mean", "std", "min", "max", "rmse"], na_rep='-',
                                          float_format=lambda x: '%.3f' % x) \
                .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm") \
                .replace("mean", "Mean (m)").replace("std", "STD. (m)").replace("min", "min (m)")\
                .replace("max", "max (m)").replace("rmse", "rmse (m)")
            with open(save_path, "w") as fp:
                fp.write(odom_res_tex)
        except Exception as e:
            logger.error(f"Could not save tex files")
            logger.exception(e)

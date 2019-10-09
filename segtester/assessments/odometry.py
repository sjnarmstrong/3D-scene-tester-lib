from typing import TYPE_CHECKING
from matplotlib import pyplot as plt
from tqdm import tqdm
from segtester import logger
from segtester.metrics.odo_ape_rpe import ape, rpe, create_plots
from evo.tools import file_interface
from evo.core.metrics import Unit
import os


ALIGNMENT_OPTIONS = {
    "noalign": (False, False, False),
    "SE3_Umeyama_alignment": (True, False, False),
    "Sim3_Umeyama_alignment": (True, True, False),
    "scale_corrected": (False, True, False),
    "origin_alignment": (False, False, True),
}


class PlotOptions:
    def __init__(self):
        self.plot_colormap_min = None
        self.plot_colormap_max = None
        self.plot_colormap_max_percentile = None
        self.plot = False
        self.serialize_plot = False


DEFAULT_PLOT_OPTION = PlotOptions()


if TYPE_CHECKING:
    from segtester.configs.assessments.odometry import OdometryConfig


class OdometryAssessment:
    def __init__(self, conf: 'OdometryConfig'):
        self.conf = conf

    def __call__(self, base_result_path, gt_dataset_conf, est_dataset_conf, *_, **__):
        est_dataset = est_dataset_conf.get_dataset()
        gt_dataset = gt_dataset_conf.get_dataset()

        gt_poses_map = {}

        for scene in tqdm(est_dataset.scenes, desc="scene"):
            try:
                est_traj = scene.get_pose_path()
                if est_traj is None:
                    logger.warn(f"{scene.alg_name}: {scene.id} - Does not have a est_traj. Skiping...")
                    continue

                if scene.id not in gt_poses_map:
                    gt_poses_map[scene.id] = gt_dataset.get_scene_with_id(scene.id).get_pose_path()
                gt_traj = gt_poses_map[scene.id]

                for alignment_opt in self.conf.alignment_options:
                    for pose_relation in self.conf.pose_relations:
                        alignment_opt_tuple = ALIGNMENT_OPTIONS[alignment_opt]
                        result_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                            "dataset_id": gt_dataset_conf.id, "scene_id": scene.id,
                            "alg_name": scene.alg_name, "pose_relation": pose_relation.value,
                            "alignment_opt": alignment_opt,
                        }).replace(" ", "_")

                        if result_path[-1] == "/":
                            os.makedirs(result_path, exist_ok=True)
                        else:
                            os.makedirs(os.path.split(result_path)[0], exist_ok=True)

                        if self.conf.run_ape_tests:
                            self.exe_ape_tests(gt_traj.get_trajectory_copy(), est_traj.get_trajectory_copy(),
                                               alignment_opt_tuple, result_path,
                                               pose_relation, self.conf.create_plots, self.conf.confirm_overwrite)

                        if self.conf.run_rpe_tests:
                            self.exe_rpe_tests(gt_traj.get_trajectory_copy(), est_traj.get_trajectory_copy(),
                                               alignment_opt_tuple, result_path,
                                               pose_relation, self.conf.create_plots, self.conf.confirm_overwrite)

            except Exception as e:
                logger.exception(f"Exception when odometry assessment on {scene.alg_name}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                logger.error(str(e))

    @staticmethod
    def exe_ape_tests(gt_traj, t_traj, alignment_opt_tuple, result_path, pose_relation,
                      should_create_plots, confirm_overwrite):

        ape_results = ape(
            gt_traj,
            t_traj,
            pose_relation,
            align=alignment_opt_tuple[0],
            correct_scale=alignment_opt_tuple[1],
            align_origin=alignment_opt_tuple[2]
        )
        if should_create_plots:
            create_plots(
                save_name=f'{result_path}ape.pdf',
                confirm_overwrite=confirm_overwrite,
                plot_opts=DEFAULT_PLOT_OPTION,
                result=ape_results,
                traj_ref=gt_traj,
                traj_est=ape_results.trajectories["estimate"]
            )

        save_name = f'{result_path}ape.zip'
        for key, val in ape_results.stats.items():
            ape_results.stats[key] = float(val)
        file_interface.save_res_file(save_name,
                                     ape_results,
                                     confirm_overwrite=confirm_overwrite)

    @staticmethod
    def exe_rpe_tests(gt_traj, t_traj, alignment_opt_tuple, result_path, pose_relation,
                      should_create_plots, confirm_overwrite):

        rpe_results = rpe(
            gt_traj,
            t_traj,
            pose_relation,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.1,
            all_pairs=False,
            align=alignment_opt_tuple[0],
            correct_scale=alignment_opt_tuple[1],
            align_origin=alignment_opt_tuple[2],
            support_loop=True)
        if should_create_plots:
            create_plots(
                save_name=f'{result_path}rpe.pdf',
                confirm_overwrite=confirm_overwrite,
                plot_opts=DEFAULT_PLOT_OPTION,
                result=rpe_results,
                traj_ref=gt_traj,
                traj_est=rpe_results.trajectories["estimate"]
            )
        save_name = f'{result_path}rpe.zip'
        for key, val in rpe_results.stats.items():
            rpe_results.stats[key] = float(val)
        file_interface.save_res_file(save_name,
                                     rpe_results,
                                     confirm_overwrite=confirm_overwrite)
from evo.core import metrics, trajectory
from segtester import logger, SEP
from evo.tools import plot
from evo.tools.settings import SETTINGS

import matplotlib.pyplot as plt
import numpy as np


def ape(traj_ref, traj_est, pose_relation, align=False, correct_scale=False,
        align_origin=False, ref_name="reference", est_name="estimate"):

    # Align the trajectories.
    only_scale = correct_scale and not align
    if align or correct_scale:
        logger.debug(SEP)
        traj_est = trajectory.align_trajectory(traj_est, traj_ref,
                                               correct_scale, only_scale)
    elif align_origin:
        logger.debug(SEP)
        traj_est = trajectory.align_trajectory_origin(traj_est, traj_ref)

    # Calculate APE.
    logger.debug(SEP)
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    title = str(ape_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"

    ape_result = ape_metric.get_result(ref_name, est_name)
    ape_result.info["title"] = title

    logger.debug(SEP)
    logger.info(ape_result.pretty_str())

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, trajectory.PoseTrajectory3D):
        seconds_from_start = [
            t - traj_est.timestamps[0] for t in traj_est.timestamps
        ]
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)

    return ape_result


def rpe(traj_ref, traj_est, pose_relation, delta, delta_unit,
        rel_delta_tol=0.1, all_pairs=False, align=False, correct_scale=False,
        align_origin=False, ref_name="reference", est_name="estimate",
        support_loop=False):

    # Align the trajectories.
    only_scale = correct_scale and not align
    if align or correct_scale:
        logger.debug(SEP)
        traj_est = trajectory.align_trajectory(traj_est, traj_ref,
                                               correct_scale, only_scale)
    elif align_origin:
        logger.debug(SEP)
        traj_est = trajectory.align_trajectory_origin(traj_est, traj_ref)

    # Calculate RPE.
    logger.debug(SEP)
    data = (traj_ref, traj_est)
    rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, rel_delta_tol,
                             all_pairs)
    rpe_metric.process_data(data)

    title = str(rpe_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"

    rpe_result = rpe_metric.get_result(ref_name, est_name)
    rpe_result.info["title"] = title
    logger.debug(SEP)
    logger.info(rpe_result.pretty_str())

    # Restrict trajectories to delta ids for further processing steps.
    if support_loop:
        # Avoid overwriting if called repeatedly e.g. in Jupyter notebook.
        import copy
        traj_ref = copy.deepcopy(traj_ref)
        traj_est = copy.deepcopy(traj_est)
    traj_ref.reduce_to_ids(rpe_metric.delta_ids)
    traj_est.reduce_to_ids(rpe_metric.delta_ids)
    rpe_result.add_trajectory(ref_name, traj_ref)
    rpe_result.add_trajectory(est_name, traj_est)

    if isinstance(traj_est, trajectory.PoseTrajectory3D) and not all_pairs:
        seconds_from_start = [
            t - traj_est.timestamps[0] for t in traj_est.timestamps
        ]
        rpe_result.add_np_array("seconds_from_start", seconds_from_start)
        rpe_result.add_np_array("timestamps", traj_est.timestamps)

    return rpe_result


def create_plots(save_name, confirm_overwrite, plot_opts, result, traj_ref, traj_est):
    logger.debug(SEP)
    logger.debug("Plotting results... ")

    plot_collection = plot.PlotCollection(result.info["title"])

    for plot_mode in create_plots.PLOT_MODES:
        fig1 = plt.figure(figsize=SETTINGS.plot_figsize)
        #fig2.clear()

        # Plot the raw metric values.
        if "seconds_from_start" in result.np_arrays:
            seconds_from_start = result.np_arrays["seconds_from_start"]
        else:
            seconds_from_start = None

        plot.error_array(fig1, result.np_arrays["error_array"],
                         x_array=seconds_from_start, statistics=result.stats,
                         name=result.info["label"], title=result.info["title"],
                         xlabel="$t$ (s)" if seconds_from_start else "index")

        # Plot the values color-mapped onto the trajectory.
        fig2 = plt.figure(figsize=SETTINGS.plot_figsize)
        ax = plot.prepare_axis(fig2, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, style=SETTINGS.plot_reference_linestyle,
                  color=SETTINGS.plot_reference_color, label='reference',
                  alpha=SETTINGS.plot_reference_alpha)

        if plot_opts.plot_colormap_min is None:
            plot_opts.plot_colormap_min = result.stats["min"]
        if plot_opts.plot_colormap_max is None:
            plot_opts.plot_colormap_max = result.stats["max"]
        if plot_opts.plot_colormap_max_percentile is not None:
            plot_opts.plot_colormap_max = np.percentile(
                result.np_arrays["error_array"], plot_opts.plot_colormap_max_percentile)

        plot.traj_colormap(ax, traj_est, result.np_arrays["error_array"],
                           plot_mode, min_map=plot_opts.plot_colormap_min,
                           max_map=plot_opts.plot_colormap_max,
                           title="Error mapped onto trajectory")
        fig2.axes.append(ax)

        plot_collection.add_figure(plot_mode.value+"raw", fig1)
        plot_collection.add_figure(plot_mode.value+"map", fig2)

    plot_collection.export(save_name,
                           confirm_overwrite=confirm_overwrite)
    plt.close('all')


create_plots.PLOT_MODES = [plot.PlotMode.xy, plot.PlotMode.xz, plot.PlotMode.yz, plot.PlotMode.xyz]

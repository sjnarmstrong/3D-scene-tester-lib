from segtester import logger, SEP


def get_mock_args(ref, plot_mode, save_plot, pose_files=None, subcommand='kitti', plot=False, config=None):
    import argparse
    args = argparse.Namespace(
        align=False, align_origin=False, config=None, correct_scale=False, debug=False, full_check=False,
        invert_transform=False, logfile=None, merge=False, n_to_align=-1, no_warnings=False, plot=plot,
        plot_mode=plot_mode, pose_files=[] if pose_files is None else [], propagate_transform=False, ref=ref,
        save_as_bag=False, save_as_kitti=False, save_as_tum=False, save_plot=save_plot, serialize_plot=None,
        silent=False, subcommand=subcommand, sync=False, t_max_diff=0.01, t_offset=0.0, transform_left=None,
        transform_right=None, verbose=False
    )

    if config:
        merged_config_dict = vars(args).copy()
        merged_config_dict.update(config)
        args = argparse.Namespace(**merged_config_dict)
    return args


def die(msg):
    import sys
    logger.error(msg)
    sys.exit(1)


def load_trajectories(args):
    from collections import OrderedDict
    from evo.tools import file_interface
    trajectories = OrderedDict()
    ref_traj = None
    if args.subcommand == "tum":
        for traj_file in args.traj_files:
            if traj_file == args.ref:
                continue
            trajectories[traj_file] = file_interface.read_tum_trajectory_file(
                traj_file)
        if args.ref:
            ref_traj = file_interface.read_tum_trajectory_file(args.ref)
    elif args.subcommand == "kitti":
        for pose_file in args.pose_files:
            if pose_file == args.ref:
                continue
            trajectories[pose_file] = file_interface.read_kitti_poses_file(
                pose_file)
        if args.ref:
            ref_traj = file_interface.read_kitti_poses_file(args.ref)
    elif args.subcommand == "euroc":
        for csv_file in args.state_gt_csv:
            if csv_file == args.ref:
                continue
            else:
                trajectories[
                    csv_file] = file_interface.read_euroc_csv_trajectory(
                        csv_file)
        if args.ref:
            ref_traj = file_interface.read_euroc_csv_trajectory(args.ref)
    elif args.subcommand == "bag":
        if not (args.topics or args.all_topics):
            die("No topics used - specify topics or set --all_topics.")
        import rosbag
        bag = rosbag.Bag(args.bag)
        try:
            if args.all_topics:
                topics = file_interface.get_supported_topics(bag)
                if args.ref in topics:
                    topics.remove(args.ref)
                if len(topics) == 0:
                    die("No topics of supported types: {}".format(" ".join(
                        file_interface.SUPPORTED_ROS_MSGS)))
            else:
                topics = args.topics
            for topic in topics:
                if topic == args.ref:
                    continue
                trajectories[topic] = file_interface.read_bag_trajectory(
                    bag, topic)
            if args.ref:
                ref_traj = file_interface.read_bag_trajectory(bag, args.ref)
        finally:
            bag.close()
    return trajectories, ref_traj


# TODO refactor
def print_traj_info(name, traj, verbose=False, full_check=False,
                    compact_name=True):
    import os
    from evo.core import trajectory

    logger.info(SEP)
    if compact_name:
        # /some/super/long/path/that/nobody/cares/about/traj.txt  ->  traj
        name = os.path.splitext(os.path.basename(name))[0]
    logger.info("name:\t" + name)

    if verbose or full_check:
        infos = traj.get_infos()
        info_str = ""
        for info, value in sorted(infos.items()):
            info_str += "\n\t" + info + "\t" + str(value)
        logger.info("infos:" + info_str)
        if full_check:
            passed, details = traj.check()
            check_str = ""
            for test, result in sorted(details.items()):
                check_str += "\n\t" + test + "\t" + result
            logger.info("checks:" + check_str)
            stat_str = ""
            try:
                stats = traj.get_statistics()
                for stat, value in sorted(stats.items()):
                    if isinstance(value, float):
                        stat_str += "\n\t" + stat + "\t" + "{0:.6f}".format(
                            value)
                    else:
                        stat_str += value
            except trajectory.TrajectoryException as e:
                stat_str += "\n\terror - " + str(e)
            logger.info("stats:" + stat_str)
    else:
        logger.info("infos:\t" + str(traj))


def run(args, trajectories, ref_traj):
    import os

    import evo.core.lie_algebra as lie
    from evo.core import trajectory
    from evo.tools import file_interface
    from evo.tools.settings import SETTINGS

    if args.debug:
        import pprint
        logger.debug("main_parser config:\n" + pprint.pformat(
            {arg: getattr(args, arg)
             for arg in vars(args)}) + "\n")
    logger.debug(SEP)

    if args.merge:
        if args.subcommand == "kitti":
            die("Can't merge KITTI files.")
        if len(trajectories) == 0:
            die("No trajectories to merge (excluding --ref).")
        trajectories = {
            "merged_trajectory": trajectory.merge(trajectories.values())
        }

    if args.transform_left or args.transform_right:
        tf_type = "left" if args.transform_left else "right"
        tf_path = args.transform_left \
                if args.transform_left else args.transform_right
        transform = file_interface.load_transform_json(tf_path)
        logger.debug(SEP)
        if not lie.is_se3(transform):
            logger.warning("Not a valid SE(3) transformation!")
        if args.invert_transform:
            transform = lie.se3_inverse(transform)
        logger.debug("Applying a {}-multiplicative transformation:\n{}".format(
            tf_type, transform))
        for traj in trajectories.values():
            traj.transform(transform, right_mul=args.transform_right,
                           propagate=args.propagate_transform)

    if args.t_offset:
        logger.debug(SEP)
        for name, traj in trajectories.items():
            if type(traj) is trajectory.PosePath3D:
                die("{} doesn't have timestamps - can't add time offset.".
                    format(name))
            logger.info("Adding time offset to {}: {} (s)".format(
                name, args.t_offset))
            traj.timestamps += args.t_offset

    if args.n_to_align != -1 and not (args.align or args.correct_scale):
        die("--n_to_align is useless without --align or/and --correct_scale")

    if args.sync or args.align or args.correct_scale or args.align_origin:
        from evo.core import sync
        if not args.ref:
            logger.debug(SEP)
            die("Can't align or sync without a reference! (--ref)  *grunt*")
        for name, traj in trajectories.items():
            if args.subcommand == "kitti":
                ref_traj_tmp = ref_traj
            else:
                logger.debug(SEP)
                ref_traj_tmp, trajectories[name] = sync.associate_trajectories(
                    ref_traj, traj, max_diff=args.t_max_diff,
                    first_name="reference", snd_name=name)
            if args.align or args.correct_scale:
                logger.debug(SEP)
                logger.debug("Aligning {} to reference.".format(name))
                trajectories[name] = trajectory.align_trajectory(
                    trajectories[name], ref_traj_tmp,
                    correct_scale=args.correct_scale,
                    correct_only_scale=args.correct_scale and not args.align,
                    n=args.n_to_align)
            if args.align_origin:
                logger.debug(SEP)
                logger.debug("Aligning {}'s origin to reference.".format(name))
                trajectories[name] = trajectory.align_trajectory_origin(
                    trajectories[name], ref_traj_tmp)

    print_compact_name = not args.subcommand == "bag"
    for name, traj in trajectories.items():
        print_traj_info(name, traj, args.verbose, args.full_check,
                        print_compact_name)
    if args.ref:
        print_traj_info(args.ref, ref_traj, args.verbose, args.full_check,
                        print_compact_name)

    if args.plot or args.save_plot or args.serialize_plot:
        import numpy as np
        from evo.tools import plot
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
        fig_xyz, axarr_xyz = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
        fig_rpy, axarr_rpy = plt.subplots(3, sharex="col",
                                          figsize=tuple(SETTINGS.plot_figsize))
        fig_traj = plt.figure(figsize=tuple(SETTINGS.plot_figsize))

        plot_mode = plot.PlotMode[args.plot_mode]
        ax_traj = plot.prepare_axis(fig_traj, plot_mode)

        if args.ref:
            short_traj_name = os.path.splitext(os.path.basename(args.ref))[0]
            if SETTINGS.plot_usetex:
                short_traj_name = short_traj_name.replace("_", "\\_")
            plot.traj(ax_traj, plot_mode, ref_traj,
                      style=SETTINGS.plot_reference_linestyle,
                      color=SETTINGS.plot_reference_color,
                      label=short_traj_name,
                      alpha=SETTINGS.plot_reference_alpha)
            plot.traj_xyz(
                axarr_xyz, ref_traj, style=SETTINGS.plot_reference_linestyle,
                color=SETTINGS.plot_reference_color, label=short_traj_name,
                alpha=SETTINGS.plot_reference_alpha)
            plot.traj_rpy(
                axarr_rpy, ref_traj, style=SETTINGS.plot_reference_linestyle,
                color=SETTINGS.plot_reference_color, label=short_traj_name,
                alpha=SETTINGS.plot_reference_alpha)

        cmap_colors = None
        if SETTINGS.plot_multi_cmap.lower() != "none":
            cmap = getattr(cm, SETTINGS.plot_multi_cmap)
            cmap_colors = iter(cmap(np.linspace(0, 1, len(trajectories))))

        for name, traj in trajectories.items():
            if cmap_colors is None:
                color = next(ax_traj._get_lines.prop_cycler)['color']
            else:
                color = next(cmap_colors)
            if print_compact_name:
                short_traj_name = os.path.splitext(os.path.basename(name))[0]
            else:
                short_traj_name = name
            if SETTINGS.plot_usetex:
                short_traj_name = short_traj_name.replace("_", "\\_")
            plot.traj(ax_traj, plot_mode, traj, '-', color, short_traj_name,
                      alpha=SETTINGS.plot_trajectory_alpha)
            if args.ref and isinstance(ref_traj, trajectory.PoseTrajectory3D):
                start_time = ref_traj.timestamps[0]
            else:
                start_time = None
            plot.traj_xyz(axarr_xyz, traj, '-', color, short_traj_name,
                          alpha=SETTINGS.plot_trajectory_alpha,
                          start_timestamp=start_time)
            plot.traj_rpy(axarr_rpy, traj, '-', color, short_traj_name,
                          alpha=SETTINGS.plot_trajectory_alpha,
                          start_timestamp=start_time)
            fig_rpy.text(
                0., 0.005, "euler_angle_sequence: {}".format(
                    SETTINGS.euler_angle_sequence), fontsize=6)

        plot_collection.add_figure("trajectories", fig_traj)
        plot_collection.add_figure("xyz_view", fig_xyz)
        plot_collection.add_figure("rpy_view", fig_rpy)
        if args.plot:
            plot_collection.show()
        if args.save_plot:
            logger.info(SEP)
            plot_collection.export(args.save_plot,
                                   confirm_overwrite=not args.no_warnings)
        if args.serialize_plot:
            logger.info(SEP)
            plot_collection.serialize(args.serialize_plot,
                                      confirm_overwrite=not args.no_warnings)

    if args.save_as_tum:
        logger.info(SEP)
        for name, traj in trajectories.items():
            dest = os.path.splitext(os.path.basename(name))[0] + ".tum"
            file_interface.write_tum_trajectory_file(
                dest, traj, confirm_overwrite=not args.no_warnings)
        if args.ref:
            dest = os.path.splitext(os.path.basename(args.ref))[0] + ".tum"
            file_interface.write_tum_trajectory_file(
                dest, ref_traj, confirm_overwrite=not args.no_warnings)
    if args.save_as_kitti:
        logger.info(SEP)
        for name, traj in trajectories.items():
            dest = os.path.splitext(os.path.basename(name))[0] + ".kitti"
            file_interface.write_kitti_poses_file(
                dest, traj, confirm_overwrite=not args.no_warnings)
        if args.ref:
            dest = os.path.splitext(os.path.basename(args.ref))[0] + ".kitti"
            file_interface.write_kitti_poses_file(
                dest, ref_traj, confirm_overwrite=not args.no_warnings)
    if args.save_as_bag:
        import datetime
        import rosbag
        dest_bag_path = str(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + ".bag"
        logger.info(SEP)
        logger.info("Saving trajectories to " + dest_bag_path + "...")
        bag = rosbag.Bag(dest_bag_path, 'w')
        try:
            for name, traj in trajectories.items():
                dest_topic = os.path.splitext(os.path.basename(name))[0]
                frame_id = traj.meta[
                    "frame_id"] if "frame_id" in traj.meta else ""
                file_interface.write_bag_trajectory(bag, traj, dest_topic,
                                                    frame_id)
            if args.ref:
                dest_topic = os.path.splitext(os.path.basename(args.ref))[0]
                frame_id = ref_traj.meta[
                    "frame_id"] if "frame_id" in ref_traj.meta else ""
                file_interface.write_bag_trajectory(bag, ref_traj, dest_topic,
                                                    frame_id)
        finally:
            bag.close()


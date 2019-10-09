from segtester import logger, SEP


def get_mock_args(save_plot, save_table, result_files, config=None):
    import argparse
    args = argparse.Namespace(config=None, debug=False, ignore_title=False, logfile=None, merge=False, no_warnings=False,
                              plot=False, plot_markers=False, result_files=result_files,
                              save_plot=save_plot, save_table=save_table, serialize_plot=None,
                              silent=False, use_filenames=False, use_rel_time=False, verbose=False)

    if config:
        merged_config_dict = vars(args).copy()
        merged_config_dict.update(config)
        args = argparse.Namespace(**merged_config_dict)
    return args


def load_results_as_dataframe(result_files, use_filenames=False, merge=False):
    import pandas as pd
    from evo.tools import pandas_bridge
    from evo.tools import file_interface

    if merge:
        from evo.core.result import merge_results
        results = [file_interface.load_res_file(f) for f in result_files]
        return pandas_bridge.result_to_df(merge_results(results))

    df = pd.DataFrame()
    for result_file in result_files:
        result = file_interface.load_res_file(result_file)
        name = result_file if use_filenames else None
        df = pd.concat([df, pandas_bridge.result_to_df(result, name)],
                       axis="columns")
    return df


def run(args, names):
    import sys

    import pandas as pd

    from evo.tools import log, user, settings
    from evo.tools.settings import SETTINGS

    pd.options.display.width = 80
    pd.options.display.max_colwidth = 20

    log.configure_logging(args.verbose, args.silent, args.debug,
                          local_logfile=args.logfile)
    if args.debug:
        import pprint
        arg_dict = {arg: getattr(args, arg) for arg in vars(args)}
        logger.debug("main_parser config:\n{}\n".format(
            pprint.pformat(arg_dict)))

    df = load_results_as_dataframe(args.result_files, args.use_filenames,
                                   args.merge)

    df.columns = names
    keys = df.columns.values.tolist()
    if SETTINGS.plot_usetex:
        keys = [key.replace("_", "\\_") for key in keys]
        df.columns = keys
    duplicates = [x for x in keys if keys.count(x) > 1]
    if duplicates:
        logger.error("Values of 'est_name' must be unique - duplicates: {}\n"
                     "Try using the --use_filenames option to use filenames "
                     "for labeling instead.".format(", ".join(duplicates)))
        sys.exit(1)

    # derive a common index type if possible - preferably timestamps
    common_index = None
    time_indices = ["timestamps", "seconds_from_start", "sec_from_start"]
    if args.use_rel_time:
        del time_indices[0]
    for idx in time_indices:
        if idx not in df.loc["np_arrays"].index:
            continue
        if df.loc["np_arrays", idx].isnull().values.any():
            continue
        else:
            common_index = idx
            break

    # build error_df (raw values) according to common_index
    if common_index is None:
        # use a non-timestamp index
        error_df = pd.DataFrame(df.loc["np_arrays", "error_array"].tolist(),
                                index=keys).T
    else:
        error_df = pd.DataFrame()
        for key in keys:
            new_error_df = pd.DataFrame({
                key: df.loc["np_arrays", "error_array"][key]
            }, index=df.loc["np_arrays", common_index][key])
            duplicates = new_error_df.index.duplicated(keep="first")
            if any(duplicates):
                logger.warning(
                    "duplicate indices in error array of {} - "
                    "keeping only first occurrence of duplicates".format(key))
                new_error_df = new_error_df[~duplicates]
            error_df = pd.concat([error_df, new_error_df], axis=1)

    # check titles
    first_title = df.loc["info", "title"][0] if not args.ignore_title else ""
    first_file = args.result_files[0]
    if not args.no_warnings and not args.ignore_title:
        checks = df.loc["info", "title"] != first_title
        for i, differs in enumerate(checks):
            if not differs:
                continue
            else:
                mismatching_title = df.loc["info", "title"][i]
                mismatching_file = args.result_files[i]
                logger.debug(SEP)
                logger.warning(
                    CONFLICT_TEMPLATE.format(first_file, first_title,
                                             mismatching_title,
                                             mismatching_file))
                if not user.confirm(
                        "You can use --ignore_title to just aggregate data.\n"
                        "Go on anyway? - enter 'y' or any other key to exit"):
                    sys.exit()

    logger.debug(SEP)
    logger.debug("Aggregated dataframe:\n{}".format(
        df.to_string(line_width=80)))

    # show a statistics overview
    logger.debug(SEP)
    if not args.ignore_title:
        logger.info("\n" + first_title + "\n\n")
    logger.info(df.loc["stats"].T.to_string(line_width=80) + "\n")

    if args.save_table:
        logger.debug(SEP)
        if args.no_warnings or user.check_and_confirm_overwrite(
                args.save_table):
            if SETTINGS.table_export_data.lower() == "error_array":
                data = error_df
            elif SETTINGS.table_export_data.lower() in ("info", "stats"):
                data = df.loc[SETTINGS.table_export_data.lower()]
            else:
                raise ValueError(
                    "unsupported export data specifier: {}".format(
                        SETTINGS.table_export_data))
            if SETTINGS.table_export_transpose:
                data = data.T

            if SETTINGS.table_export_format == "excel":
                writer = pd.ExcelWriter(args.save_table)
                data.to_excel(writer)
                writer.save()
                writer.close()
            else:
                getattr(data,
                        "to_" + SETTINGS.table_export_format)(args.save_table)
            logger.debug("{} table saved to: {}".format(
                SETTINGS.table_export_format, args.save_table))

    if args.plot or args.save_plot or args.serialize_plot:
        # check if data has NaN "holes" due to different indices
        inconsistent = error_df.isnull().values.any()
        if inconsistent and common_index != "timestamps" and not args.no_warnings:
            logger.debug(SEP)
            logger.warning("Data lengths/indices are not consistent, "
                           "raw value plot might not be correctly aligned")

        from evo.tools import plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        import math

        # use default plot settings
        figsize = (SETTINGS.plot_figsize[0], SETTINGS.plot_figsize[1])
        use_cmap = SETTINGS.plot_multi_cmap.lower() != "none"
        colormap = SETTINGS.plot_multi_cmap if use_cmap else None
        linestyles = ["-o" for x in args.result_files
                      ] if args.plot_markers else None

        # labels according to first dataset
        if "xlabel" in df.loc["info"].index and not df.loc[
                "info", "xlabel"].isnull().values.any():
            index_label = df.loc["info", "xlabel"][0]
        else:
            index_label = "$t$ (s)" if common_index else "index"
        metric_label = df.loc["info", "label"][0]

        plot_collection = plot.PlotCollection(first_title)
        # raw value plot
        fig_raw = plt.figure(figsize=figsize)
        # handle NaNs from concat() above
        error_df.interpolate(method="index").plot(
            ax=fig_raw.gca(), colormap=colormap, style=linestyles,
            title=first_title, alpha=SETTINGS.plot_trajectory_alpha)
        plt.xlabel(index_label)
        plt.ylabel(metric_label)
        plt.legend(frameon=True)
        plot_collection.add_figure("raw", fig_raw)

        # statistics plot
        plot_statistics = ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse']
        if plot_statistics:
            fig_stats = plt.figure(figsize=figsize)
            include = df.loc["stats"].index.isin(plot_statistics)
            if any(include):
                df.loc["stats"][include].plot(kind="barh", ax=fig_stats.gca(),
                                              colormap=colormap, stacked=False)
                plt.xlabel(metric_label)
                plt.legend(frameon=True)
                plot_collection.add_figure("stats", fig_stats)

        # grid of distribution plots
        raw_tidy = pd.melt(error_df, value_vars=list(error_df.columns.values),
                           var_name="estimate", value_name=metric_label)
        col_wrap = 2 if len(args.result_files) <= 2 else math.ceil(
            len(args.result_files) / 2.0)
        dist_grid = sns.FacetGrid(raw_tidy, col="estimate", col_wrap=col_wrap)
        # TODO: see issue #98
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist_grid.map(sns.distplot, metric_label)  # fits=stats.gamma
        plot_collection.add_figure("histogram", dist_grid.fig)

        # box plot
        fig_box = plt.figure(figsize=figsize)
        ax = sns.boxplot(x=raw_tidy["estimate"], y=raw_tidy[metric_label],
                         ax=fig_box.gca())
        # ax.set_xticklabels(labels=[item.get_text() for item in ax.get_xticklabels()], rotation=30)
        plot_collection.add_figure("box_plot", fig_box)

        # violin plot
        fig_violin = plt.figure(figsize=figsize)
        ax = sns.violinplot(x=raw_tidy["estimate"], y=raw_tidy[metric_label],
                            ax=fig_violin.gca())
        # ax.set_xticklabels(labels=[item.get_text() for item in ax.get_xticklabels()], rotation=30)
        plot_collection.add_figure("violin_histogram", fig_violin)

        if args.plot:
            plot_collection.show()
        if args.save_plot:
            logger.debug(SEP)
            plot_collection.export(args.save_plot,
                                   confirm_overwrite=not args.no_warnings)
        if args.serialize_plot:
            logger.debug(SEP)
            plot_collection.serialize(args.serialize_plot,
                                      confirm_overwrite=not args.no_warnings)


if __name__ == '__main__':
    from evo import entry_points
    entry_points.res()

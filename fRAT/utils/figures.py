import itertools
import os
import warnings

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import plotnine as pltn
from matplotlib import pyplot as plt
from nilearn import plotting

from .utils import Utils

config = None


class Figures:
    @classmethod
    def Make_figures(cls, cfg):
        global config
        config = cfg

        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        plt.rcParams['figure.figsize'] = config.brain_table_x_size, config.brain_table_y_size  # Brain plot code

        combined_results_df = pd.read_json("combined_results.json")

        if config.multicore_processing & (config.make_one_region_fig or config.make_histogram):
            pool = Utils.start_processing_pool()
        else:
            pool = None

        if config.make_brain_table:
            brain_plot_opts_exts = ["_Mean.nii.gz", "_Mean_within_roi_scaled.nii.gz", "_Mean_mixed_roi_scaled.nii.gz",
                                    "all"]
            brain_plot_base_ext = brain_plot_opts_exts[config.brain_fig_file]

            if brain_plot_base_ext == "all":
                for base_extension in brain_plot_opts_exts[0:-1]:
                    cls.brain_facet_grid(combined_results_df, base_extension)
            else:
                cls.brain_facet_grid(combined_results_df, brain_plot_base_ext)

        if config.make_scatter_table:
            cls.scatter_plot(combined_results_df)

        if config.make_one_region_fig:
            cls.one_region_bar_chart_setup(combined_results_df, pool)

        if config.make_histogram:
            cls.region_histogram_setup(combined_results_df, pool)

        if pool:
            pool.close()
            pool.join()

    @classmethod
    def scatter_plot(cls, df):
        Utils.check_and_make_dir("Figures/Scatterplots")
        df = df[(df['index'] != 'Overall') & (df['index'] != 'No ROI')]  # Remove No ROI and Overall rows

        df = df.groupby([config.table_cols, config.table_rows]).apply(
            lambda x: x.sort_values(['Mean']))  # Group by parameters and sort
        df = df.reset_index(drop=True)  # Reset index to remove grouping

        scatterplots = ['roi_ordered', 'stat_ordered']
        if config.table_row_order == 'roi':
            scatterplots.remove('stat')
        elif config.table_row_order == 'statorder':
            scatterplots.remove('roi_ordered')

        for scatterplot in scatterplots:
            if config.verbose:
                print(f"Preparing {scatterplot} scatterplot!")

            if scatterplot == 'roi_ordered':
                roi_ord = pd.Categorical(df['index'],
                                         categories=df['index'].unique())  # Order rows based on first facet
            else:
                roi_ord = pd.Categorical(df.groupby(['MB', 'SENSE']).cumcount())  # Order each facet individually

            figure_table = (pltn.ggplot(df, pltn.aes(x="Mean", y=roi_ord))
                            + pltn.geom_point(na_rm=True, size=1)
                            + pltn.geom_errorbarh(pltn.aes(xmin="Mean-Conf_Int_95", xmax="Mean+Conf_Int_95"),
                                                  na_rm=True, height=None)
                            + pltn.scale_y_discrete(labels=[])
                            + pltn.ylab(config.table_y_label)
                            + pltn.xlab(config.table_x_label)
                            + pltn.facet_grid('{rows}~{cols}'.format(rows=config.table_rows, cols=config.table_cols),
                                              drop=True, labeller="label_both")
                            + pltn.theme_538()  # Set theme
                            + pltn.theme(panel_grid_major_y=pltn.themes.element_line(alpha=0),
                                         panel_grid_major_x=pltn.themes.element_line(alpha=1),
                                         panel_background=pltn.element_rect(fill="gray", alpha=0.1),
                                         dpi=config.plot_dpi))

            figure_table.save(f"Figures/Scatterplots/{scatterplot}_scatterplot.png", height=config.plot_scale,
                              width=config.plot_scale * 3,
                              verbose=False, limitsize=False)
            if config.verbose:
                print("Saved scatterplot!")

    @classmethod
    def one_region_bar_chart_setup(cls, df, pool):
        Utils.check_and_make_dir("Figures/Barcharts")
        list_rois = list(df['index'].unique())

        chosen_rois = cls.find_chosen_rois(list_rois, plot_name="One region bar chart",
                                           config_region_var=config.single_roi_fig_regions)

        iterable = zip(itertools.repeat(Figures.one_region_bar_chart_make), chosen_rois,
                       itertools.repeat(df), itertools.repeat(list_rois), itertools.repeat(config))

        if pool:
            pool.starmap(Utils.class_method_handler, iterable)
        else:
            list(itertools.starmap(Utils.class_method_handler, iterable))

    @staticmethod
    def one_region_bar_chart_make(roi, df, list_rois, config):
        thisroi = list_rois[roi]

        current_df = df.loc[df['index'] == thisroi]

        current_df = current_df.sort_values([config.single_roi_fig_x_axis])
        current_df = current_df.reset_index(drop=True)  # Reset index to remove grouping
        current_df[config.single_roi_fig_x_axis] = pd.Categorical(current_df[config.single_roi_fig_x_axis],
                                                                  categories=current_df[
                                                                      config.single_roi_fig_x_axis].unique())

        figure = (
                pltn.ggplot(current_df, pltn.aes(x=config.single_roi_fig_x_axis, y='Mean',
                                                 ymin="Mean-Conf_Int_95", ymax="Mean+Conf_Int_95",
                                                 fill='factor({colour})'.format(
                                                     colour=config.single_roi_fig_colour)))
                + pltn.theme_538()
                + pltn.geom_col(position=pltn.position_dodge(preserve='single', width=0.8), width=0.8, na_rm=True)
                + pltn.geom_errorbar(size=1, position=pltn.position_dodge(preserve='single', width=0.8))
                + pltn.labs(x=config.single_roi_fig_label_x, y=config.single_roi_fig_label_y,
                            fill=config.single_roi_fig_label_fill)
                + pltn.scale_x_discrete(labels=[])
                + pltn.theme(panel_grid_major_x=pltn.element_line(alpha=0),
                             axis_title_x=pltn.element_text(weight='bold', color='black', size=20),
                             axis_title_y=pltn.element_text(weight='bold', color='black', size=20),
                             axis_text_y=pltn.element_text(size=20, color='black'),
                             legend_title=pltn.element_text(size=20, color='black'),
                             legend_text=pltn.element_text(size=18, color='black'),
                             subplots_adjust={'right': 0.85},
                             legend_position=(0.9, 0.8),
                             dpi=config.plot_dpi
                             )
                + pltn.geom_text(pltn.aes(y=-.7, label='MB'),  # TODO make MB label variable
                                 color='black', size=20, va='top')
                + pltn.scale_fill_manual(values=config.colorblind_friendly_plot_colours)
        )

        figure.save("Figures/Barcharts/{thisroi}_barplot.png".format(thisroi=thisroi), height=config.plot_scale,
                    width=config.plot_scale * 3,
                    verbose=False, limitsize=False)

        if config.verbose:
            print("Saved {thisroi}_barplot.png".format(thisroi=thisroi))

    @classmethod
    def region_histogram_setup(cls, combined_df, pool):
        Utils.check_and_make_dir("Figures/Histograms")
        list_rois = list(combined_df['index'].unique())
        chosen_rois = cls.find_chosen_rois(list_rois, plot_name="Histogram",
                                           config_region_var=config.histogram_fig_regions)

        # Compile a dataframe containing raw values and parameter values for all ROIs and save as combined_raw_df
        if chosen_rois:
            jsons = Utils.find_files("Raw_results", "json")
            combined_raw_df = cls.make_raw_df(jsons, combined_df)

            if config.verbose:
                print(f"STAGE 1 -- Dataframe setup")
            combined_raw_dfs = []
            for roi in chosen_rois:
                combined_raw_dfs.append(cls.region_histogram_df_make(roi, combined_raw_df, list_rois, config))

            if config.verbose:
                print(f"STAGE 2 -- Histogram creation")
            iterable = zip(itertools.repeat(Figures.region_histogram_make), chosen_rois,
                           combined_raw_dfs, itertools.repeat(list_rois), itertools.repeat(config))

            if pool:
                pool.starmap(Utils.class_method_handler, iterable)
            else:
                list(itertools.starmap(Utils.class_method_handler, iterable))

    @staticmethod
    def region_histogram_df_make(roi, combined_raw_df, list_rois, config):
        # Set up the df for each chosen roi
        thisroi = list_rois[roi]

        if thisroi == "No ROI":
            return pd.DataFrame()
        elif thisroi == "Overall":
            current_df = combined_raw_df.copy()
        else:
            current_df = combined_raw_df[combined_raw_df["ROI"] == thisroi].copy()

        current_df = current_df.dropna()  # Drop na values using pandas function, which is faster than plotnines dropna functions

        current_df['Mean'] = current_df.groupby([config.histogram_fig_x_facet, config.histogram_fig_y_facet])[
            "voxel_value"].transform('mean')
        current_df['Median'] = current_df.groupby([config.histogram_fig_x_facet, config.histogram_fig_y_facet])[
            "voxel_value"].transform('median')

        current_df = pd.melt(current_df, id_vars=current_df.keys()[:-2], var_name="Statistic",
                             value_vars=["Mean", "Median"], value_name="stat_value")  # Put df into correct format

        if config.histogram_show_mean and not config.histogram_show_median:
            current_df = current_df.loc[current_df["Statistic"] == "Mean"]
        elif config.histogram_show_median and not config.histogram_show_mean:
            current_df = current_df.loc[current_df["Statistic"] == "Median"]

        return current_df

    @staticmethod
    def region_histogram_make(roi, combined_raw_df, list_rois, config):
        if combined_raw_df.empty:
            return
        else:
            thisroi = list_rois[roi]

            figure = (
                    pltn.ggplot(combined_raw_df, pltn.aes(x="voxel_value"))
                    + pltn.theme_538()
                    + pltn.geom_histogram(binwidth=config.histogram_binwidth, fill=config.histogram_fig_colour)
                    + pltn.facet_grid(f"{config.histogram_fig_y_facet}~{config.histogram_fig_x_facet}",
                                      drop=True, labeller="label_both")
                    + pltn.labs(x=config.histogram_fig_label_x, y=config.histogram_fig_label_y)
                    + pltn.theme(
                panel_grid_minor_x=pltn.themes.element_line(alpha=0),
                panel_grid_major_x=pltn.themes.element_line(alpha=1),
                panel_grid_major_y=pltn.element_line(alpha=0),
                plot_background=pltn.element_rect(fill="white"),
                panel_background=pltn.element_rect(fill="gray", alpha=0.1),
                axis_title_x=pltn.element_text(weight='bold', color='black', size=20),
                axis_title_y=pltn.element_text(weight='bold', color='black', size=20),
                strip_text_x=pltn.element_text(weight='bold', size=10, color='black'),
                strip_text_y=pltn.element_text(weight='bold', size=10, color='black'),
                axis_text_x=pltn.element_text(size=10, color='black'),
                axis_text_y=pltn.element_text(size=10, color='black'),
                dpi=config.plot_dpi
                )
            )

            # Display mean or median as vertical lines on plot
            if config.histogram_show_mean or config.histogram_show_median:
                figure += pltn.geom_vline(pltn.aes(xintercept="stat_value", color="Statistic"),
                                          size=config.histogram_stat_line_size)
                figure += pltn.scale_color_manual(values=[config.colorblind_friendly_plot_colours[3],
                                                          config.colorblind_friendly_plot_colours[1]])

            # Display legend for mean and median
            if not config.histogram_show_legend:
                figure += pltn.theme(legend_position='none')

            # Suppress Pandas warning about alignment of non-concatenation axis
            warnings.simplefilter(action='ignore', category=FutureWarning)

            figure.save(f"Figures/Histograms/{thisroi}_histogram.png",
                        height=config.plot_scale, width=config.plot_scale * 3,
                        verbose=False, limitsize=False)

            warnings.simplefilter(action='default', category=FutureWarning)

            if config.verbose:
                print(f"Saved {thisroi}_histogram.png")

    @classmethod
    def brain_facet_grid(cls, df, base_extension):
        Utils.check_and_make_dir("Figures/Brain_grids")
        base_ext_clean = os.path.splitext(os.path.splitext(base_extension)[0])[0][1:]
        if config.verbose:
            print(f"Preparing {base_ext_clean} table!")
        json_array = df['File_name'].unique()

        plot_values, axis_titles, current_params, col_nums, \
        row_nums, cell_nums, y_axis_size, x_axis_size = cls.table_setup(df)

        if base_extension != "_Mean.nii.gz":
            config.brain_fig_value_max = 100

        for file_num, json in enumerate(json_array):

            # Save brain image using nilearn
            image_name = json + ".png"
            plot = plotting.plot_anat(json + base_extension,
                                      draw_cross=False, annotate=False, colorbar=True, display_mode='xz',
                                      vmin=config.brain_fig_value_min, vmax=config.brain_fig_value_max,
                                      cut_coords=(config.brain_x_coord, config.brain_z_coord),
                                      cmap='inferno')

            plot.savefig(image_name)
            plot.close()

            # Import saved image into subplot
            img = mpimg.imread(json + ".png")
            plt.subplot(y_axis_size, x_axis_size, cell_nums[file_num] + 1)
            plt.imshow(img)

            ax = plt.gca()
            ax.set_yticks([])  # Remove y-axis ticks
            ax.axes.yaxis.set_ticklabels([])  # Remove y-axis labels

            ax.set_xticks([])  # Remove x-axis ticks
            ax.axes.xaxis.set_ticklabels([])  # Remove x-axis labels

            if row_nums[file_num] == 0:
                plt.title(axis_titles[0] + " " + plot_values[0][col_nums[file_num]], fontsize=config.plot_font_size)

            if col_nums[file_num] == 0:
                plt.ylabel(axis_titles[1] + " " + plot_values[1][row_nums[file_num]],
                           fontsize=config.plot_font_size)

        cls.label_blank_cell_axes(plot_values, axis_titles, cell_nums, x_axis_size, y_axis_size)

        if config.brain_tight_layout:
            plt.tight_layout()
        plt.savefig(f"Figures/Brain_grids/{base_ext_clean}.png", dpi=config.plot_dpi, bbox_inches='tight')
        plt.close()
        if config.verbose:
            print("Saved table!")

    @classmethod
    def find_chosen_rois(cls, all_rois, plot_name, config_region_var):
        if config_region_var == 'Runtime':  # If no ROI has been selected for this plot
            chosen_rois = []
            print("\n")

            for roi_num, roi in enumerate(all_rois):
                print("{roi_num}: {roi}".format(roi_num=roi_num, roi=roi))

            while not chosen_rois:
                print(f'\n--- {plot_name} creation ---')
                roi_ans = input(
                    "Type a comma-separated list of the ROIs (listed above) you want to produce a figure for, "
                    "'e.g. 2, 15, 7, 23' or 'all' for all rois. \nAlternatively press enter to skip this step: ")

                if roi_ans.lower() == "all":
                    chosen_rois = list(range(0, len(all_rois)))

                elif len(roi_ans) > 0:
                    chosen_rois = [x.strip() for x in roi_ans.split(',')]  # Split by comma and whitespace

                    try:
                        chosen_rois = list(map(int, chosen_rois))  # Convert each list item to integers
                    except ValueError:
                        print('Comma-separated list contains non integers.\n')
                        chosen_rois = []

                else:  # Else statement for blank input, this skips creating this plot
                    chosen_rois = []
                    break

        else:  # Else if an ROI selection has been made, convert it into the correct format
            if isinstance(config_region_var, str) and config_region_var.lower() == "all":
                chosen_rois = list(range(0, len(all_rois)))
            else:
                chosen_rois = config_region_var

                if isinstance(chosen_rois, int):
                    chosen_rois = [chosen_rois]
                else:
                    chosen_rois = list(chosen_rois)

        return chosen_rois

    @staticmethod
    def make_raw_df(jsons, combined_df):
        combined_raw_df = pd.DataFrame()
        for json_file in jsons:
            current_json = pd.read_json(f"{os.getcwd()}/Raw_results/{json_file}")

            json_file_name = json_file.rsplit("_raw.json")[0]

            current_json["File_name"] = json_file_name

            # Find parameter values for each file_name
            combined_df_search = combined_df.loc[combined_df["File_name"] == json_file_name]
            current_json[config.histogram_fig_x_facet] = combined_df_search[config.histogram_fig_x_facet].iloc[0]
            current_json[config.histogram_fig_y_facet] = combined_df_search[config.histogram_fig_y_facet].iloc[0]

            combined_raw_df = combined_raw_df.append(current_json)

        combined_raw_df = combined_raw_df.melt(
            id_vars=[config.histogram_fig_x_facet, config.histogram_fig_y_facet, "File_name"],
            var_name='ROI', value_name='voxel_value')

        return combined_raw_df

    @classmethod
    def table_setup(cls, df):
        unique_params = []
        current_params = []
        col_nums = []
        row_nums = []
        cell_nums = []

        for key in config.parameter_dict:
            params = sorted(list(df[key].unique()))  # Sort list of parameters
            params = [str(param) for param in params]  # Convert parameters to strings
            unique_params.append(params)

        plot_values = unique_params  # Get axis values
        axis_titles = list(config.parameter_dict.keys())  # Get axis titles
        plot_values_sorted = [plot_values[axis_titles.index(config.brain_table_cols)],  # Sort axis values
                              plot_values[axis_titles.index(config.brain_table_rows)]]

        x_axis_size = len(plot_values_sorted[0])
        y_axis_size = len(plot_values_sorted[1])

        for file_num, file_name in enumerate(df['File_name'].unique()):
            temp_param_store = []
            file_name_row = df[df['File_name'] == file_name].iloc[0]  # Get the first row of the relevant file name

            temp_param_store.append(str(file_name_row[config.brain_table_cols]))
            temp_param_store.append(str(file_name_row[config.brain_table_rows]))

            current_params.append(temp_param_store)  # Store parameters used for file

            col_nums.append(plot_values_sorted[0].index(current_params[file_num][0]))  # Work out col number
            row_nums.append(plot_values_sorted[1].index(current_params[file_num][1]))  # Work out row number

            cell_nums.append(np.ravel_multi_index((row_nums[file_num], col_nums[file_num]),
                                                  (y_axis_size, x_axis_size)))  # Find linear index of figure

        return plot_values_sorted, axis_titles, current_params, col_nums, row_nums, cell_nums, y_axis_size, x_axis_size

    @classmethod
    def label_blank_cell_axes(cls, plot_values, axis_titles, cell_nums, x_axis_size, y_axis_size):
        for counter, x_title in enumerate(plot_values[0]):
            hidden_cell = np.ravel_multi_index((0, counter), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.title(axis_titles[0] + " " + x_title, fontsize=config.plot_font_size)

                if hidden_cell == 0:
                    plt.ylabel(axis_titles[1] + " " + plot_values[1][0], fontsize=config.plot_font_size)

                cls.make_cell_invisible()

        for counter, y_title in enumerate(plot_values[1]):
            hidden_cell = np.ravel_multi_index((counter, 0), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums and hidden_cell != 0:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.ylabel(axis_titles[1] + " " + y_title, fontsize=config.plot_font_size)

                cls.make_cell_invisible()

    @staticmethod
    def make_cell_invisible():
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        frame.spines['left'].set_visible(False)
        frame.spines['right'].set_visible(False)
        frame.spines['bottom'].set_visible(False)
        frame.spines['top'].set_visible(False)
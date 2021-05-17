import itertools
import os
import re
import warnings

import matplotlib.image as mpimg
import numpy as np
import nibabel as nib
import pandas as pd
import plotnine as pltn
import simplejson as json
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from nilearn import plotting
from PIL import Image

from .utils import Utils

config = None


class Figures:
    @classmethod
    def Make_figures(cls, cfg):
        global config
        config = cfg

        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        combined_results_df = pd.read_json("Summarised_results/combined_results.json")

        if config.multicore_processing & (config.make_one_region_fig or config.make_histogram):
            pool = Utils.start_processing_pool()
        else:
            pool = None

        if config.make_brain_table:
            brain_plot_opts_exts = ["_Mean.nii.gz",
                                    "_Mean_within_roi_scaled.nii.gz",
                                    "_Mean_mixed_roi_scaled.nii.gz",
                                    "all"]
            brain_plot_base_ext = brain_plot_opts_exts[config.brain_fig_file]

            indiv_brains_dir = f"{os.getcwd()}/Figures/Brain_images"
            Utils.check_and_make_dir(indiv_brains_dir)

            if brain_plot_base_ext == "all":
                for base_extension in brain_plot_opts_exts[0:-1]:
                    indiv_brain_imgs = cls.brain_grid(combined_results_df, base_extension)

                    for img in indiv_brain_imgs:
                        Utils.move_file(img, os.getcwd(), indiv_brains_dir)
            else:
                indiv_brain_imgs = cls.brain_grid(combined_results_df, brain_plot_base_ext)

                for img in indiv_brain_imgs:
                    Utils.move_file(img, os.getcwd(), indiv_brains_dir)

        if config.make_violin_plot:
            cls.violin_plot(combined_results_df)

        if config.make_one_region_fig:
            cls.barchart_setup(combined_results_df, pool)

        if config.make_histogram:
            cls.histogram_setup(combined_results_df, pool)

        if pool:
            pool.close()
            pool.join()

    @classmethod
    def scatterplot(cls, df):
        """Deprecated"""
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
                print(f"Saving {scatterplot} scatterplot!")

            if scatterplot == 'roi_ordered':
                roi_ord = pd.Categorical(df['index'],
                                         categories=df['index'].unique())  # Order rows based on first facet
            else:
                roi_ord = pd.Categorical(
                    df.groupby([config.table_cols, config.table_rows]).cumcount())  # Order each facet individually

            figure_table = (pltn.ggplot(df, pltn.aes(x="Mean", y=roi_ord))
                            + pltn.geom_point(na_rm=True, size=1)
                            + pltn.geom_errorbarh(pltn.aes(xmin="Mean-Conf_Int_95", xmax="Mean+Conf_Int_95"),
                                                  na_rm=True, height=None)
                            + pltn.xlim(0, None)
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

    @classmethod
    def barchart_setup(cls, df, pool):
        Utils.check_and_make_dir("Figures/Barcharts")
        list_rois = list(df['index'].unique())

        chosen_rois = cls.find_chosen_rois(list_rois, plot_name="One region bar chart",
                                           config_region_var=config.regional_fig_rois)

        if config.verbose:
            print(f'\n--- Barchart creation ---')

        ylim = 0
        while True:
            iterable = zip(itertools.repeat(Figures.barchart_make), chosen_rois,
                           itertools.repeat(df), itertools.repeat(list_rois),
                           itertools.repeat(config), itertools.repeat(ylim),
                           itertools.repeat(cls.figure_save), itertools.repeat(cls.find_axis_limit))

            if pool:
                ylim = pool.starmap(Utils.class_method_handler, iterable)

            else:
                ylim = list(itertools.starmap(Utils.class_method_handler, iterable))

            if any(ylim):
                ylim.sort(key=lambda x: x[1])

                if config.verbose:
                    print(f'Maximum y limit of: {round(ylim[-1][1])} seen with ROI: {ylim[-1][2]}. '
                          f'Creating figures with this y limit.\n')

                ylim = ylim[-1][1]
            else:
                break

    @staticmethod
    def barchart_make(roi, df, list_rois, config, ylimit, save_function, find_ylim_function):
        thisroi = list_rois[roi]

        current_df = df.loc[df['index'] == thisroi]

        current_df = current_df.sort_values([config.single_roi_fig_x_axis])
        current_df = current_df.reset_index(drop=True)  # Reset index to remove grouping
        current_df[config.single_roi_fig_x_axis] = pd.Categorical(current_df[config.single_roi_fig_x_axis],
                                                                  categories=current_df[
                                                                      config.single_roi_fig_x_axis].unique())

        current_df.columns = [c.replace(' ', '_') for c in current_df.columns]
        config.single_roi_fig_x_axis = config.single_roi_fig_x_axis.replace(" ", "_")  # TODO: Comment this
        config.single_roi_fig_colour = config.single_roi_fig_colour.replace(" ", "_")

        figure = (
                pltn.ggplot(current_df, pltn.aes(x=config.single_roi_fig_x_axis, y='Mean',
                                                 ymin="Mean-Conf_Int_95", ymax="Mean+Conf_Int_95",
                                                 fill=f'factor({config.single_roi_fig_colour})'))
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
                + pltn.geom_text(pltn.aes(y=-.7, label=config.single_roi_fig_x_axis),
                                 color='black', size=20, va='top')
                + pltn.scale_fill_manual(values=config.colorblind_friendly_plot_colours)
        )

        if ylimit:
            # Set y limit of figure (used to make it the same for every barchart)
            figure += pltn.ylim(None, ylimit)
            thisroi += '_same_ylim'

        returned_ylim = 0
        if config.use_same_axis_limits in ('Same limits', 'Create both') and ylimit == 0:
            returned_ylim = find_ylim_function(thisroi, figure, 'yaxis')

        if config.use_same_axis_limits == 'Same limits' and ylimit == 0:
            return returned_ylim
        elif ylimit != 0:
            folder = 'Same_yaxis'
        else:
            folder = 'Different_yaxis'

        save_function(figure, thisroi, config, folder, 'barchart')

        return returned_ylim

    @classmethod
    def histogram_setup(cls, combined_df, pool):
        Utils.check_and_make_dir("Figures/Histograms")
        list_rois = list(combined_df['index'].unique())
        chosen_rois = cls.find_chosen_rois(list_rois, plot_name="Histogram",
                                           config_region_var=config.regional_fig_rois)

        # Compile a dataframe containing raw values and parameter values for all ROIs and save as combined_raw_df
        if chosen_rois:
            if config.verbose:
                print(f'\n--- Histogram creation ---')

            jsons = Utils.find_files("Raw_results", "json")
            combined_raw_df = cls.make_raw_df(jsons, combined_df)

            combined_raw_dfs = []
            for roi in chosen_rois:
                combined_raw_dfs.append(cls.histogram_df_setup(roi, combined_raw_df, combined_df, list_rois, config))

            xlim = 0
            while True:
                iterable = zip(itertools.repeat(Figures.histogram_make), chosen_rois,
                               combined_raw_dfs, itertools.repeat(list_rois),
                               itertools.repeat(config), itertools.repeat(xlim),
                               itertools.repeat(cls.figure_save), itertools.repeat(cls.find_axis_limit))

                if pool:
                    xlim = pool.starmap(Utils.class_method_handler, iterable)
                else:
                    xlim = list(itertools.starmap(Utils.class_method_handler, iterable))

                if any(xlim):
                    try:
                        xlim.remove(None)
                    except ValueError:
                        pass

                    xlim.sort(key=lambda x: x[1])

                    if config.verbose:
                        print(f'Maximum x limit of: {round(xlim[-1][1])} seen with ROI: {xlim[-1][2]}. '
                              f'Creating figures with this x limit.\n')

                    xlim = xlim[-1][1]
                else:
                    break

    @staticmethod
    def histogram_df_setup(roi, combined_raw_df, combined_df, list_rois, config):
        # Set up the df for each chosen roi
        thisroi = list_rois[roi]

        if thisroi == "No ROI":
            return pd.DataFrame()
        elif thisroi == "Overall":
            current_df = combined_raw_df.copy()
            current_df['ROI'] = "Overall"
        else:
            current_df = combined_raw_df[combined_raw_df["ROI"] == thisroi].copy()

        current_df = current_df.dropna()  # Drop na values using pandas function, which is faster than plotnines dropna functions

        # Combine both dataframes to find mean and median statistics
        combined_df = combined_df.rename(
            columns={"index": "ROI"})  # Rename column to maintain parity with combined_df column naming convention

        current_df = current_df.merge(combined_df,
                                      on=['ROI', config.histogram_fig_x_facet, config.histogram_fig_y_facet],
                                      how='left')

        # Keep only the necessary columns
        keys = [config.histogram_fig_x_facet, config.histogram_fig_y_facet,
                'ROI', 'voxel_value', 'Voxels', 'Mean', 'Median']

        for column in current_df.columns:
            if column not in keys:
                current_df = current_df.drop(columns=column)

        current_df = pd.melt(current_df, id_vars=keys[:-2], var_name="Statistic",
                             value_vars=["Mean", "Median"], value_name="stat_value")  # Put df into tidy format

        if config.histogram_show_mean and not config.histogram_show_median:
            current_df = current_df.loc[current_df["Statistic"] == "Mean"]
        elif config.histogram_show_median and not config.histogram_show_mean:
            current_df = current_df.loc[current_df["Statistic"] == "Median"]

        return current_df

    @staticmethod
    def histogram_make(roi, combined_raw_df, list_rois, config, xlimit, save_function, find_xlim_function):
        if combined_raw_df.empty:
            if config.verbose:
                print('INFO: Histograms cannot be made for the No ROI category.')
            return
        else:
            thisroi = list_rois[roi]

            figure = (
                    pltn.ggplot(combined_raw_df, pltn.aes(x="voxel_value"))
                    + pltn.theme_538()
                    + pltn.geom_histogram(binwidth=config.histogram_binwidth, fill=config.histogram_fig_colour,
                                          boundary=0,
                                          na_rm=True)  # Boundary centers the bars, na_rm cancels error from setting an xlimit
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

            if xlimit:
                # Set y limit of figure (used to make it the same for every barchart)
                figure += pltn.xlim(-1, xlimit)
                thisroi += '_same_xlim'
            else:
                figure += pltn.xlim(-1, None)

            returned_xlim = 0
            if config.use_same_axis_limits in ('Same limits', 'Create both') and xlimit == 0:
                returned_xlim = find_xlim_function(thisroi, figure, 'xaxis')

            if config.use_same_axis_limits == 'Same limits' and xlimit == 0:
                return returned_xlim
            elif xlimit != 0:
                folder = 'Same_xaxis'
            else:
                folder = 'Different_xaxis'

            # Suppress Pandas warning about alignment of non-concatenation axis
            warnings.simplefilter(action='ignore', category=FutureWarning)

            save_function(figure, thisroi, config, folder, 'histogram')

            warnings.simplefilter(action='default', category=FutureWarning)

            return returned_xlim

    @classmethod
    def violin_plot(cls, df):
        Utils.check_and_make_dir("Figures/Violin_plots")
        df = df[(df['index'] != 'Overall') & (df['index'] != 'No ROI')]  # Remove No ROI and Overall rows

        df = df.groupby([config.table_cols, config.table_rows]).apply(
            lambda x: x.sort_values(['Mean']))  # Group by parameters and sort
        df = df.reset_index(drop=True)  # Reset index to remove grouping

        if config.verbose:
            print(f"Saving violin plot!")

        df['constant'] = 1

        # How much to shift the violin, points and lines
        shift = 0.1

        right_shift = pltn.aes(x=pltn.stage('constant', after_scale='x+shift'))  # shift outward
        left_shift = pltn.aes(x=pltn.stage('constant', after_scale='x-shift'))  # shift inward

        figure = (pltn.ggplot(df, pltn.aes(x="constant", y="Mean"))
                  + pltn.geom_violin(left_shift, na_rm=True, style='left', fill=config.violin_colour, size=0.6)
                  + pltn.geom_boxplot(width=0.1, outlier_alpha=0, fill=config.boxplot_colour, size=0.6)
                  + pltn.xlim(0.4, 1.4)
                  + pltn.ylab(config.table_x_label)
                  + pltn.xlab("")
                  + pltn.facet_grid('{rows}~{cols}'.format(rows=config.table_rows, cols=config.table_cols),
                                    drop=True, labeller="label_both")
                  + pltn.theme_538()  # Set theme
                  + pltn.theme(panel_grid_major_y=pltn.themes.element_line(alpha=1),
                               panel_grid_major_x=pltn.themes.element_line(alpha=0),
                               panel_background=pltn.element_rect(fill="gray", alpha=0.1),
                               axis_text_x=pltn.element_blank(),
                               dpi=config.plot_dpi))

        if config.violin_show_data:
            if config.violin_jitter:
                figure += pltn.geom_jitter(width=0.04, height=0)
            else:
                figure += pltn.geom_point()

        figure.save(f"Figures/Violin_plots/violinplot.png", height=config.plot_scale,
                    width=config.plot_scale * 3,
                    verbose=False, limitsize=False)

    @staticmethod
    def find_axis_limit(thisroi, figure, axis):
        # Find what the current axis limits will be
        fig = figure.draw()
        axes = fig.axes

        if axis == 'xaxis':
            lim = axes[0].get_xlim()

        elif axis == 'yaxis':
            lim = axes[0].get_ylim()

        lim = (*lim, thisroi)

        return lim

    @staticmethod
    def figure_save(figure, thisroi, config, folder, chart_type):

        # Format file name correctly
        replacements = [
            (r'\([^()]*\)', ""),  # Remove anything between parenthesis
            (r'[^a-zA-Z\d:]', "_"),  # Remove non-alphanumeric characters
            (r'_{2,}', "_")  # Replace multiple underscores with one
        ]
        thisroi = thisroi.replace("\'", "")  # Remove apostrophes
        for old, new in replacements:
            thisroi = re.sub(old, new, thisroi)

        Utils.check_and_make_dir(f"Figures/{chart_type.title()}s/{folder}/")
        figure.save(f"Figures/{chart_type.title()}s/{folder}/{thisroi}_{chart_type}.png",
                    height=config.plot_scale, width=config.plot_scale * 3,
                    verbose=False, limitsize=False)

        if config.verbose:
            print(f"Saved {thisroi}_{chart_type}.png")

    @classmethod
    def brain_grid(cls, df, base_extension):
        Utils.check_and_make_dir("Figures/Brain_grids")
        base_ext_clean = os.path.splitext(os.path.splitext(base_extension)[0])[0][1:]
        indiv_brain_imgs = []

        json_array = df['File_name'].unique()

        plot_values, axis_titles, current_params, col_nums, \
        row_nums, cell_nums, y_axis_size, x_axis_size = cls.table_setup(
            df)  # TODO: check if axis titles still need to be returned

        if x_axis_size in (1, 2):
            brain_table_x_size = 32
        else:
            brain_table_x_size = x_axis_size * 16

        brain_table_y_size = 15

        plt.rcParams['figure.figsize'] = brain_table_x_size, brain_table_y_size
        plt.rcParams['figure.subplot.wspace'] = 0.01
        plt.rcParams['figure.subplot.hspace'] = 0.1

        vmax = vmax_storage = config.brain_fig_value_max
        if vmax_storage is None:
            vmax_storage = []

        while True:
            if base_extension != "_Mean.nii.gz":
                vmax = 100
            elif base_extension == "_Mean.nii.gz" and vmax is not None:
                base_ext_clean += "_same_scale"

            if config.verbose:
                print(f"Saving {base_ext_clean} table!")

            for file_num, json in enumerate(json_array):
                brain_img, indiv_brain_imgs, dims = cls.create_indiv_brain_img(json, base_ext_clean, base_extension,
                                                                               vmax, vmax_storage, indiv_brain_imgs)

                # Import saved image into subplot
                img = mpimg.imread(brain_img)
                plt.subplot(y_axis_size, x_axis_size, cell_nums[file_num] + 1)
                plt.imshow(img)

                ax = plt.gca()
                ax.set_yticks([])  # Remove y-axis ticks
                ax.axes.yaxis.set_ticklabels([])  # Remove y-axis labels

                ax.set_xticks([])  # Remove x-axis ticks
                ax.axes.xaxis.set_ticklabels([])  # Remove x-axis labels

                if row_nums[file_num] == 0:
                    plt.title(config.brain_table_col_labels + " " + plot_values[0][col_nums[file_num]],
                              fontsize=config.plot_font_size)

                if col_nums[file_num] == 0:
                    plt.ylabel(config.brain_table_row_labels + " " + plot_values[1][row_nums[file_num]],
                               fontsize=config.plot_font_size)

            cls.label_blank_cell_axes(plot_values, axis_titles, cell_nums, x_axis_size, y_axis_size, dims)

            if config.brain_tight_layout:
                plt.tight_layout()

            plt.savefig(f"Figures/Brain_grids/{base_ext_clean}.png", dpi=config.plot_dpi, bbox_inches='tight')
            plt.close()

            if base_extension != "_Mean.nii.gz" or config.brain_fig_value_max is not None:
                break
            else:
                # Find highest ROI value seen to create figures with the same scale
                vmax_storage = sorted(vmax_storage)[-1]
                vmax = config.brain_fig_value_max = vmax_storage[0]

                print(f'Maximum ROI value of: {round(vmax_storage[0])} seen in file: {vmax_storage[1]}. '
                      f'Creating figures with this colourbar limit.')

        return indiv_brain_imgs

    @classmethod
    def create_indiv_brain_img(cls, json, base_ext_clean, base_extension, vmax, vmax_storage, indiv_brain_imgs):
        # Save brain image using nilearn
        brain_img = f"{json}_{base_ext_clean}.png"
        indiv_brain_imgs.append(brain_img)

        if base_extension == "_Mean.nii.gz" and base_ext_clean.find("_same_scale") == -1:
            # Calculate colour bar limit if not manually set
            brain = nib.load(f"NIFTI_ROI/{json}{base_extension}")
            brain = brain.get_fdata()

            vmax = np.nanmax(brain)
            vmax_storage.append((np.nanmax(brain), json))  # Save vmax to find highest vmax later

        plot = plotting.plot_anat(f"NIFTI_ROI/{json}{base_extension}",
                                  draw_cross=False, annotate=False, colorbar=True, display_mode='xz',
                                  vmin=config.brain_fig_value_min, vmax=vmax,
                                  cut_coords=(config.brain_x_coord, config.brain_z_coord),
                                  cmap='inferno')
        plot.savefig(brain_img)
        plot.close()

        im = Image.open(brain_img)
        width, height = im.size

        return brain_img, indiv_brain_imgs, (width, height)

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
            if isinstance(config_region_var, list) and isinstance(config_region_var[0], str) \
                    and config_region_var[0].lower() == "all":
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
            with open(f"{os.getcwd()}/Raw_results/{json_file}", 'r') as f:
                current_json = Utils.dict_to_dataframe(json.load(f))

            json_file_name = json_file.rsplit("_raw.json")[0]

            current_json["File_name"] = json_file_name

            # Find parameter values for each file_name
            combined_df_search = combined_df.loc[combined_df["File_name"] == json_file_name]

            try:
                current_json[config.histogram_fig_x_facet] = combined_df_search[config.histogram_fig_x_facet].iloc[0]
                current_json[config.histogram_fig_y_facet] = combined_df_search[config.histogram_fig_y_facet].iloc[0]
            except IndexError:
                continue

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
    def label_blank_cell_axes(cls, plot_values, axis_titles, cell_nums, x_axis_size, y_axis_size, dims):
        # Make blank image with the same dimensions as previous images
        img = Image.new("RGB", (dims[0], dims[1]), (255, 255, 255))

        for counter, x_title in enumerate(plot_values[0]):
            hidden_cell = np.ravel_multi_index((0, counter), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.imshow(img)
                cls.make_cell_invisible()

                plt.title(config.brain_table_col_labels + " " + x_title, fontsize=config.plot_font_size)

                if hidden_cell == 0:
                    plt.ylabel(config.brain_table_row_labels + " " + plot_values[1][0], fontsize=config.plot_font_size)

        for counter, y_title in enumerate(plot_values[1]):
            hidden_cell = np.ravel_multi_index((counter, 0), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums and hidden_cell != 0:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.imshow(img)
                cls.make_cell_invisible()

                plt.ylabel(config.brain_table_row_labels + " " + y_title, fontsize=config.plot_font_size)

    @staticmethod
    def make_cell_invisible():
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        frame.spines['left'].set_visible(False)
        frame.spines['right'].set_visible(False)
        frame.spines['bottom'].set_visible(False)
        frame.spines['top'].set_visible(False)

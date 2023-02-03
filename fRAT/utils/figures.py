import copy
import itertools
import os
import re
import warnings
from glob import glob
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import nibabel as nib
import pandas as pd
import plotnine as pltn
import simplejson as json
from nilearn import plotting
from matplotlib import pyplot as plt
from PIL import Image

from .utils import Utils


class Figures:
    config = None

    @classmethod
    def make_figures(cls, cfg, config_path, config_filename):
        cls.config = cfg

        combined_results_df, _ = Utils.read_combined_results(os.getcwd(), cls.config.averaging_type)

        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        Utils.save_config('Figures', config_path, config_filename,
                          new_config_name='figure_log',
                          relevant_sections=['Parsing',
                                             'Plotting', 'General plot settings',
                                             'Violin plot',
                                             'Brain table', 'Regional bar chart',
                                             'Regional histogram'],
                          additional_info=[f'data_used_for_figures = '
                                           f'"{cfg.averaging_type.replace(" ", "_").lower()}"'])

        if cls.config.verbose:
            print(f'\nUsing {cfg.averaging_type.replace(" ", "_").lower()} data to create figures.')

        if cls.config.multicore_processing & (cls.config.make_one_region_fig or cls.config.make_histogram):
            pool = Utils.start_processing_pool()
        else:
            pool = None

        if cls.config.make_brain_table:
            if cls.config.verbose:
                print(f'\n--- Brain grid creation ---')

            BrainGrid.make(combined_results_df)

        if cls.config.make_violin_plot:
            if cls.config.verbose:
                print(f'\n--- Violin plot creation ---')

            Utils.check_and_make_dir("Figures/Violin_plots")
            ViolinPlot.make(combined_results_df)

        if cls.config.make_one_region_fig:
            Barchart.setup(combined_results_df, pool)

        if cls.config.make_histogram:
            Histogram.setup(combined_results_df, pool)

        if pool:
            pool.close()
            pool.join()

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
    def figure_save(figure, thisroi, folder, chart_type, config):
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

        figure.save(f"Figures/{chart_type.title()}s/{folder}/{thisroi}_{chart_type}.svg",
                    height=config.plot_scale, width=config.plot_scale * 3,
                    verbose=False, limitsize=False)

        if config.verbose:
            print(f"Saved {thisroi}_{chart_type}.png")

    @staticmethod
    def load_and_restructure_jsons(config, jsons, combined_df, data_type):
        combined_raw_df = pd.DataFrame()

        # Make a list of significant columns and remove any blank values
        # TODO remove references to histogram to make it more generalisable

        if data_type == 'statistics':
            signif_columns = list(filter(None, [*config.parameter_dict1, "File_name", 'subject']))
        else:
            signif_columns = list(filter(None, [config.histogram_fig_x_facet, config.histogram_fig_y_facet,
                                                "File_name"]))

        for json_file in jsons:
            with open(json_file, 'r') as f:
                current_json = Utils.dict_to_dataframe(json.load(f))

            try:
                current_json = current_json.loc[['Mean', 'Voxels']]
            except KeyError:  # Will raise a KeyError if passing in raw results
                pass

            delimeters = ['_ps', '_raw', '.json']
            json_file_name = os.path.basename(json_file)

            for delimeter in delimeters:
                json_file_name = json_file_name.rsplit(delimeter)[0]

            # Find parameter values for each file_name
            combined_df_search = combined_df.loc[combined_df["File_name"].str.lower() == json_file_name.lower()]
            combined_df_search.columns = combined_df_search.columns.str.lower()

            try:
                for column in signif_columns:
                    current_json[column] = combined_df_search[column.lower()].iloc[0]
            except KeyError:
                pass
            except IndexError:
                continue

            if data_type == 'statistics':
                current_json['subject'] = re.findall('sub-[0-9]*', json_file)[0]

            combined_raw_df = combined_raw_df.append(current_json)

        # Convert it from wide format into long format
        mean_data = combined_raw_df.iloc[0::2].melt(id_vars=signif_columns, var_name='ROI', value_name='voxel_value')
        voxel_data = combined_raw_df.iloc[1::2].melt(id_vars=signif_columns, var_name='ROI', value_name='voxel_value')

        mean_data['voxel_amount'] = voxel_data['voxel_value']
        mean_data.dropna(inplace=True)  # Drop rows that have NA for voxel value

        return mean_data


class BrainGrid(Figures):
    @classmethod
    def make(cls, combined_results_df):
        indiv_brains_dir = f"{os.getcwd()}/Figures/Brain_images"
        Utils.check_and_make_dir(indiv_brains_dir)
        Utils.check_and_make_dir(f"{os.getcwd()}/Figures/Brain_grids")

        if cls.config.averaging_type == 'Session averaged':
            statistic_labels = cls.config.statistic_options['Session averaged']
            subfolder = 'Session_averaged_results'
        else:
            statistic_labels = cls.config.statistic_options['Participant averaged']
            subfolder = 'Participant_averaged_results'

        for statistic in statistic_labels:
            # If NIFTI files have not been created for statistic, skip creating the figure
            if not glob(f"Overall/NIFTI_ROI/{subfolder}/*{statistic}*"):
                continue

            Utils.check_and_make_dir(f"{os.getcwd()}/Figures/Brain_grids/{statistic}")

            brain_plot_exts = [f"_{statistic}.nii.gz",
                               f"_{statistic}_within_roi_scaled.nii.gz",
                               f"_{statistic}_mixed_roi_scaled.nii.gz"]

            for base_extension in brain_plot_exts:
                indiv_brain_imgs = cls.setup(combined_results_df, base_extension, statistic, subfolder)

                for img in indiv_brain_imgs:
                    Utils.move_file(img, os.getcwd(), indiv_brains_dir)

            if cls.config.verbose:
                print("\n")

    @classmethod
    def setup(cls, df, base_extension, statistic, subfolder):
        base_ext_clean = os.path.splitext(os.path.splitext(base_extension)[0])[0][1:]
        indiv_brain_imgs = []

        json_array = df['File_name'].unique()

        # Create new list of files which exist in NIFTI_ROI folder
        json_array = [jsn for jsn in json_array if glob(f"Overall/NIFTI_ROI/{subfolder}/{jsn}{base_extension}")]

        critical_params, cell_nums, y_axis_size, x_axis_size = cls.table_setup(df)

        if x_axis_size in (1, 2):
            brain_table_x_size = 32
        else:
            brain_table_x_size = x_axis_size * 16

        brain_table_y_size = 15

        plt.rcParams['figure.figsize'] = brain_table_x_size, brain_table_y_size
        plt.rcParams['figure.subplot.wspace'] = 0.01
        plt.rcParams['figure.subplot.hspace'] = 0.1

        if statistic in ['Mean', 'Median']:
            vmax = vmax_storage = cls.config.brain_fig_value_max
            vmin = cls.config.brain_fig_value_min
        else:
            vmax = vmax_storage = None
            vmin = 0

        if vmax_storage is None:
            vmax_storage = []

        while True:
            if base_extension != f"_{statistic}.nii.gz":
                # For within and mixed scaled tables
                vmax = 100
                vmin = 0
            elif base_extension == f"_{statistic}.nii.gz" and vmax is not None:
                base_ext_clean += "_same_scale"

            if cls.config.verbose:
                print(f"Saving {base_ext_clean} table.")

            indiv_brain_imgs = cls.make_table(base_ext_clean, base_extension, cell_nums, indiv_brain_imgs, json_array,
                                              critical_params, statistic, vmax, vmax_storage, vmin, x_axis_size,
                                              y_axis_size, subfolder)

            if vmax is not None:
                break

            else:
                # Find highest ROI value seen to create figures with the same scale
                vmax_storage = sorted(vmax_storage)[-1]
                vmax = vmax_storage[0]

                if cls.config.verbose:
                    print(
                        f'Maximum ROI value of: {round(vmax_storage[0])} seen for parameter combination: {vmax_storage[1]}. '
                        f'Creating figures with this colourbar limit.')

        return indiv_brain_imgs

    @classmethod
    def make_table(cls, base_ext_clean, base_extension, cell_nums, indiv_brain_imgs, json_array, critical_params,
                   statistic, vmax, vmax_storage, vmin, x_axis_size, y_axis_size, subfolder):
        for file_num, json in enumerate(json_array):
            brain_img, indiv_brain_imgs, dims = cls.save_brain_imgs(json, base_ext_clean, base_extension,
                                                                    vmax, vmax_storage, vmin, indiv_brain_imgs,
                                                                    statistic, subfolder)

            # Import saved image into subplot
            img = mpimg.imread(brain_img)
            plt.subplot(y_axis_size, x_axis_size, cell_nums[file_num] + 1)
            plt.imshow(img)

            ax = plt.gca()
            ax.set_yticks([])  # Remove y-axis ticks
            ax.axes.yaxis.set_ticklabels([])  # Remove y-axis labels

            ax.set_xticks([])  # Remove x-axis ticks
            ax.axes.xaxis.set_ticklabels([])  # Remove x-axis labels

            if critical_params['rows']['order'][file_num] == 0 and len(critical_params['cols']['values']) != 1:
                plt.title(cls.config.brain_table_col_labels + " "
                          + str(critical_params['cols']['values'][critical_params['cols']['order'][file_num]]),
                          fontsize=cls.config.plot_font_size)

            if critical_params['cols']['order'][file_num] == 0 and len(critical_params['rows']['values']) != 1:
                plt.ylabel(cls.config.brain_table_row_labels + " "
                           + str(critical_params['rows']['values'][critical_params['rows']['order'][file_num]]),
                           fontsize=cls.config.plot_font_size)

        cls.label_blank_cell_axes(critical_params, cell_nums, x_axis_size, y_axis_size, dims)

        if cls.config.brain_tight_layout:
            plt.tight_layout()

        plt.savefig(f"Figures/Brain_grids/{statistic}/{base_ext_clean}.png", dpi=cls.config.plot_dpi,
                    bbox_inches='tight')
        plt.savefig(f"Figures/Brain_grids/{statistic}/{base_ext_clean}.svg", dpi=cls.config.plot_dpi,
                    bbox_inches='tight')
        plt.close()

        return indiv_brain_imgs

    @classmethod
    def save_brain_imgs(cls, json, base_ext_clean, base_extension, vmax, vmax_storage, vmin, indiv_brain_imgs,
                        statistic, subfolder):
        # Save brain image using nilearn
        png_path = f"{json}_{base_ext_clean}.png"
        indiv_brain_imgs.append(png_path)

        nifti_path = f"Overall/NIFTI_ROI/{subfolder}/{json}{base_extension}"

        if base_extension == f"_{statistic}.nii.gz" and base_ext_clean.find("_same_scale") == -1:
            # Calculate colour bar limit if not manually set
            brain = nib.load(nifti_path)
            brain = brain.get_fdata()

            vmax = np.nanmax(brain)

            # BE CAREFUL: changes to the vmax_storage list applies outside of the function.
            # If this isn't changed later it is definitely due to efficiency not laziness.
            vmax_storage.append((np.nanmax(brain), json))  # Save vmax to find highest vmax later

        # Correct error caused by nan values by converting them to zeros
        img = nib.load(nifti_path)
        img = nib.Nifti1Image(np.nan_to_num(img.get_fdata()), img.affine, img.header)

        plot = plotting.plot_anat(img,
                                  draw_cross=False, annotate=False, colorbar=True, display_mode='xz',
                                  vmin=vmin, vmax=vmax,
                                  cut_coords=(cls.config.brain_x_coord, cls.config.brain_z_coord),
                                  cmap='inferno')
        plot.savefig(png_path, dpi=cls.config.plot_dpi)
        plot.close()

        im = Image.open(png_path)
        width, height = im.size

        return png_path, indiv_brain_imgs, (width, height)

    @classmethod
    def table_setup(cls, df):
        unique_params = []
        cell_nums = []

        for key in cls.config.parameter_dict:
            params = sorted(list(df[key].unique()))  # Sort list of parameters
            params = [str(param) for param in params]  # Convert parameters to strings
            unique_params.append(params)

        plot_values = unique_params  # Get axis values
        axis_titles = list(cls.config.parameter_dict.keys())  # Get axis titles

        critical_params = {'cols': {'param': cls.config.brain_table_cols, 'values': [0], 'order': []},
                           'rows': {'param': cls.config.brain_table_rows, 'values': [0], 'order': []}}

        for axis in critical_params:
            if critical_params[axis]['param'] == '':
                continue
            else:
                critical_params[axis]['values'] = plot_values[
                    axis_titles.index(critical_params[axis]['param'])]  # Sort axis values

        x_axis_size = len(critical_params['cols']['values'])
        y_axis_size = len(critical_params['rows']['values'])

        for file_num, file_name in enumerate(df['File_name'].unique()):
            temp_order_store = []
            file_name_row = df[df['File_name'] == file_name].iloc[0]  # Get the first row of the relevant file name

            for axis in critical_params:
                if critical_params[axis]['values'] != [0]:
                    # Extract parameter from df for file
                    file_param = str(file_name_row[critical_params[axis]['param']])
                    # Work out row/col order
                    critical_params[axis]['order'].append(critical_params[axis]['values'].index(file_param))
                    temp_order_store.append(critical_params[axis]['values'].index(file_param))
                else:
                    critical_params[axis]['order'].append(0)
                    temp_order_store.append(0)

            cell_nums.append(np.ravel_multi_index((temp_order_store[1], temp_order_store[0]),
                                                  (y_axis_size, x_axis_size)))  # Find linear index of figure

        return critical_params, cell_nums, y_axis_size, x_axis_size

    @classmethod
    def label_blank_cell_axes(cls, critical_params, cell_nums, x_axis_size, y_axis_size, dims):
        # Make blank image with the same dimensions as previous images
        img = Image.new("RGB", (dims[0], dims[1]), (255, 255, 255))

        for counter, x_title in enumerate(critical_params['cols']['values']):
            hidden_cell = np.ravel_multi_index((0, counter), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.imshow(img)
                cls.make_cell_invisible()

                plt.title(cls.config.brain_table_col_labels + " " + x_title, fontsize=cls.config.plot_font_size)

                if hidden_cell == 0:
                    plt.ylabel(cls.config.brain_table_row_labels + " " + critical_params['rows']['values'][0],
                               fontsize=cls.config.plot_font_size)

        for counter, y_title in enumerate(critical_params['rows']['values']):
            hidden_cell = np.ravel_multi_index((counter, 0), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums and hidden_cell != 0:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.imshow(img)
                cls.make_cell_invisible()

                plt.ylabel(cls.config.brain_table_row_labels + " " + y_title, fontsize=cls.config.plot_font_size)

    @staticmethod
    def make_cell_invisible():
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        frame.spines['left'].set_visible(False)
        frame.spines['right'].set_visible(False)
        frame.spines['bottom'].set_visible(False)
        frame.spines['top'].set_visible(False)


class ViolinPlot(Figures):
    @classmethod
    def make(cls, orig_df):
        df = copy.deepcopy(orig_df)

        df = df[(df['index'] != 'Overall') & (df['index'] != 'No ROI')]  # Remove No ROI and Overall rows
        df = df[df['Mean'] > 0]  # Remove any ROIs with value of 0, indicating a failure to fit

        if not cls.config.table_cols == '' and not cls.config.table_rows == '':
            # If parameter to plot for both columns and rows is set then group by both parameters and then sort by mean
            df = df.groupby([cls.config.table_cols, cls.config.table_rows]).apply(lambda x: x.sort_values(['Mean']))

        elif not cls.config.table_cols == '':
            df = df.groupby(cls.config.table_cols).apply(lambda x: x.sort_values(['Mean']))

        elif not cls.config.table_rows == '':
            df = df.groupby(cls.config.table_rows).apply(lambda x: x.sort_values(['Mean']))

        df = df.reset_index(drop=True)  # Reset index to remove grouping

        df['constant'] = 1

        # How much to shift the violin, points and lines
        shift = 0.1

        right_shift = pltn.aes(x=pltn.stage('constant', after_scale='x+shift'))  # shift outward
        left_shift = pltn.aes(x=pltn.stage('constant', after_scale='x-shift'))  # shift inward

        figure = (pltn.ggplot(df)
                  + pltn.aes(x="constant", y="Mean")
                  + pltn.geom_violin(left_shift, na_rm=True, style='left', fill=cls.config.violin_colour, size=0.6)
                  + pltn.geom_boxplot(width=0.1, outlier_alpha=0, fill=cls.config.boxplot_colour, size=0.6)
                  + pltn.xlim(0.4, 1.4)
                  + pltn.ylim(0, None)
                  + pltn.ylab(cls.config.table_x_label)
                  + pltn.xlab("")
                  + pltn.facet_grid(f'{cls.config.table_rows}~{cls.config.table_cols}', drop=True,
                                    labeller="label_both")
                  + pltn.theme_538()  # Set theme
                  + pltn.theme(panel_grid_major_y=pltn.themes.element_line(alpha=1),
                               panel_grid_major_x=pltn.themes.element_line(alpha=0),
                               panel_background=pltn.element_rect(fill="gray", alpha=0.1),
                               axis_text_x=pltn.element_blank(),
                               dpi=cls.config.plot_dpi))

        if cls.config.violin_show_data:
            if cls.config.violin_jitter:
                figure += pltn.geom_jitter(width=0.04, height=0)
            else:
                figure += pltn.geom_point()

        figure.save(f"Figures/Violin_plots/violinplot.png", height=cls.config.plot_scale,
                    width=cls.config.plot_scale * 3,
                    verbose=False, limitsize=False)

        figure.save(f"Figures/Violin_plots/violinplot.svg", height=cls.config.plot_scale,
                    width=cls.config.plot_scale * 3,
                    verbose=False, limitsize=False)

        if cls.config.verbose:
            print(f"Saved violin plot!")


class Barchart(Figures):
    @classmethod
    def setup(cls, df, pool):
        if cls.config.single_roi_fig_x_axis == '':
            raise Exception('Parameter to plot along the x-axes of the barcharts has not been set.')

        Utils.check_and_make_dir("Figures/Barcharts")

        list_rois = list(df['index'].unique())
        chosen_rois = Utils.find_chosen_rois(list_rois, func_name="One region bar chart",
                                             config_region_var=cls.config.regional_fig_rois)
        try:
            chosen_rois
        except NameError:
            if cls.config.verbose:
                print('Skipping barchart creation.')
        else:
            if cls.config.verbose:
                print(f'\n--- Barchart creation ---')

            ylim = 0
            while True:
                iterable = zip(itertools.repeat(cls.make), chosen_rois, itertools.repeat(df),
                               itertools.repeat(ylim), itertools.repeat(cls.figure_save),
                               itertools.repeat(cls.find_axis_limit), itertools.repeat(cls.config))

                if pool:
                    ylim = pool.starmap(Utils.class_method_handler, iterable)

                else:
                    ylim = list(itertools.starmap(Utils.class_method_handler, iterable))

                if any(ylim):
                    ylim.sort(key=lambda x: x[1])

                    if cls.config.verbose:
                        print(f'Maximum y limit of: {round(ylim[-1][1])} seen with ROI: {ylim[-1][2]}. '
                              f'Creating figures with this y limit.\n')

                    ylim = ylim[-1][1]
                else:
                    break

    @staticmethod
    def make(thisroi, df, ylimit, save_function, find_ylim_function, config):
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
                pltn.ggplot(current_df)
                + pltn.theme_538()
                + pltn.geom_col(position=pltn.position_dodge(preserve='single', width=0.8), width=0.8, na_rm=True)
                + pltn.geom_errorbar(size=1, position=pltn.position_dodge(preserve='single', width=0.8))
                + pltn.scale_x_discrete(labels=[])
                + pltn.theme(panel_grid_major_x=pltn.element_line(alpha=0),
                             panel_background=pltn.element_rect(fill='white', alpha=.2),
                             axis_title_x=pltn.element_text(weight='bold', color='black', size=20),
                             axis_title_y=pltn.element_text(weight='bold', color='black', size=20),
                             axis_text_y=pltn.element_text(size=20, color='black'),
                             legend_title=pltn.element_text(size=20, weight='bold', color='black', margin={'b': 20}),
                             legend_text=pltn.element_text(size=20, color='black', margin={'l': 5}),
                             legend_entry_spacing=10,
                             legend_key_size=30,
                             subplots_adjust={'right': 0.85},
                             legend_position=(0.9, 0.8),
                             dpi=config.plot_dpi
                             )
                + pltn.geom_text(pltn.aes(y=-.7, label=config.single_roi_fig_x_axis),
                                 color='black', size=20, va='top')
                + pltn.scale_fill_manual(values=config.colorblind_friendly_plot_colours)
        )

        conf_int_string = [x for x in current_df.keys() if 'Conf_Int' in x][0]

        if not config.single_roi_fig_colour == '':
            figure += pltn.aes(x=config.single_roi_fig_x_axis, y='Mean',
                               ymin=f"Mean-{conf_int_string}", ymax=f"Mean+{conf_int_string}",
                               fill=f'factor({config.single_roi_fig_colour})')
        else:
            figure += pltn.aes(x=config.single_roi_fig_x_axis, y='Mean',
                               ymin=f"Mean-{conf_int_string}", ymax=f"Mean+{conf_int_string}")

        figure += pltn.labs(x=config.single_roi_fig_label_x, y=config.single_roi_fig_label_y,
                            fill=config.single_roi_fig_label_fill)

        if ylimit:
            # Set y limit of figure (used to make it the same for every barchart)
            figure += pltn.ylim(None, ylimit)
            thisroi += '_same_ylim'

        returned_ylim = 0
        if ylimit == 0:
            returned_ylim = find_ylim_function(thisroi, figure, 'yaxis')
            folder = 'Different_yaxis'
        elif ylimit != 0:
            folder = 'Same_yaxis'

        save_function(figure, thisroi, folder, 'barchart', config)

        return returned_ylim


class Histogram(Figures):
    @classmethod
    def setup(cls, combined_df, pool):
        Utils.check_and_make_dir("Figures/Histograms")

        list_rois = list(combined_df['index'].unique())
        chosen_rois = Utils.find_chosen_rois(list_rois, func_name="Histogram",
                                             config_region_var=cls.config.regional_fig_rois)

        try:
            chosen_rois
        except NameError:
            if cls.config.verbose:
                print('Skipping histogram creation.')
        else:
            if cls.config.verbose:
                print(f'\n--- Histogram creation ---')

            # Compile a dataframe containing raw values and parameter values for all ROIs and save as combined_raw_df
            jsons = [f"{os.getcwd()}/Overall/Raw_results/{jsn}" for jsn in
                     Utils.find_files("Overall/Raw_results", "json")]
            combined_raw_df = cls.load_and_restructure_jsons(cls.config, jsons, combined_df, data_type='plotting')

            combined_raw_dfs = []
            for roi in chosen_rois:
                combined_raw_dfs.append(
                    cls.get_mean_and_median_for_each_chosen_roi(roi, combined_raw_df, combined_df))

            xlim = 0
            while True:
                iterable = zip(itertools.repeat(cls.make), chosen_rois, combined_raw_dfs,
                               itertools.repeat(xlim), itertools.repeat(cls.figure_save),
                               itertools.repeat(cls.find_axis_limit), itertools.repeat(cls.config))

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

                    if cls.config.verbose:
                        print(f'Maximum x limit of: {round(xlim[-1][1])} seen with ROI: {xlim[-1][2]}. '
                              f'Creating figures with this x limit.\n')

                    xlim = xlim[-1][1]
                else:
                    break

    @classmethod
    def get_mean_and_median_for_each_chosen_roi(cls, thisroi, combined_raw_df, combined_df):
        # Set up the df for each chosen roi

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

        # Make a list of significant columns and remove any blank values
        signif_columns = list(filter(None, [cls.config.histogram_fig_x_facet, cls.config.histogram_fig_y_facet]))
        current_df = current_df.merge(combined_df,
                                      on=['ROI', *signif_columns],
                                      how='left')

        # Keep only the necessary columns
        keys = [*signif_columns, 'ROI', 'voxel_value', 'Mean', 'Median']

        for column in current_df.columns:
            if column not in keys:
                current_df = current_df.drop(columns=column)

        current_df = pd.melt(current_df, id_vars=keys[:-2], var_name="Statistic",
                             value_vars=["Mean", "Median"], value_name="stat_value")  # Put df into tidy format

        if cls.config.histogram_show_mean and not cls.config.histogram_show_median:
            current_df = current_df.loc[current_df["Statistic"] == "Mean"]
        elif cls.config.histogram_show_median and not cls.config.histogram_show_mean:
            current_df = current_df.loc[current_df["Statistic"] == "Median"]

        return current_df

    @staticmethod
    def make(thisroi, combined_raw_df, xlimit, save_function, find_xlim_function, config):
        if combined_raw_df.empty:
            if config.verbose:
                print('INFO: Histograms cannot be made for the No ROI category.')
            return
        else:
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
            if xlimit == 0:
                returned_xlim = find_xlim_function(thisroi, figure, 'xaxis')
                folder = 'Different_xaxis'
            elif xlimit != 0:
                folder = 'Same_xaxis'

            # Suppress Pandas warning about alignment of non-concatenation axis
            warnings.simplefilter(action='ignore', category=FutureWarning)

            save_function(figure, thisroi, folder, 'histogram', config)

            warnings.simplefilter(action='default', category=FutureWarning)

            return returned_xlim


class CompareOutputs(Figures):
    current_df = None  # TODO: See what this class could be used for

    @classmethod
    def run(cls, config):
        df, labels = cls.setup_df(config)
        cls.Make_scatter(df, labels, config)

    @classmethod
    def setup_df(cls, config):
        dfs = {}
        labels = []
        for x in range(2):
            directory = Utils.file_browser(title='Select the directory output by the fRAT')

            with open(f"{directory}/Summarised_results/combined_results.json", "r") as results:
                data = json.load(results)
                dfs[x] = pd.DataFrame(data)
                labels.append(os.path.basename(directory))

                rois = sorted({d['index'] for d in data})  # Using set returns only unique values
            # TODO: run a check to make sure same ROIs, check they have same critical parameters and same parameter space
        dfm = dfs[0].merge(dfs[1], how='outer',
                           on=['index', config.histogram_fig_y_facet, config.histogram_fig_x_facet])

        return dfm, labels

    @classmethod
    def Make_scatter(cls, df, labels, config):
        figure = (
                pltn.ggplot(df, pltn.aes(x="Mean_x", y="Mean_y"))
                + pltn.theme_538()
                + pltn.geom_point()
                + pltn.facet_grid(f"{config.histogram_fig_y_facet}~{config.histogram_fig_x_facet}",
                                  drop=True, labeller="label_both")
                + pltn.geom_smooth(method='lm')
                + pltn.labs(x=labels[1], y=labels[0])  # TODO Check these labels are the right way round
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

        figure.save(f"TEST.png", height=config.plot_scale,
                    width=config.plot_scale * 3,
                    verbose=False, limitsize=False)

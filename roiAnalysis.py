import os
import sys
import xmltodict
import json
import warnings
import re
import config

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import plotnine as pltn
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from copy import deepcopy
from scipy.sparse import csr_matrix
from glob import glob
from nipype.interfaces import fsl
from matplotlib import pyplot as plt
from nilearn import plotting
from tkinter import Tk, filedialog


class Analysis:
    brain_extract = config.brain_extract
    frac_inten = config.frac_inten
    dof = config.dof

    atlas_number = config.atlas_number
    conf_level_number = config.conf_level_number
    roi_stat_number = config.roi_stat_number

    include_rois = config.include_rois
    exclude_rois = config.exclude_rois

    save_stats_only = config.save_stats_only
    run_analysis = config.run_analysis
    stat_map_folder = config.stat_map_folder
    stat_map_suffix = config.stat_map_suffix
    bootstrap = config.bootstrap

    _roi_stat_list = ["Voxel number", "Mean", "Standard Deviation", "Confidence Interval", "Min", "Max"]
    _conf_level_list = [('80', 1.28),
                        ('85', 1.44),
                        ('90', 1.64),
                        ('95', 1.96),
                        ('98', 2.33),
                        ('99', 2.58)]
    _atlas_label_list = [('Cerebellum/Cerebellum-MNIflirt-maxprob-thr0-1mm.nii.gz', 'Cerebellum_MNIflirt.xml'),
                         ('Cerebellum/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz', 'Cerebellum_MNIfnirt.xml'),
                         ('HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz', 'HarvardOxford-Cortical.xml'),
                         ('HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz', 'HarvardOxford-Subcortical.xml'),
                         ('JHU/JHU-ICBM-labels-1mm.nii.gz', 'JHU-labels.xml'),
                         ('JHU/JHU-ICBM-tracts-maxprob-thr0-1mm.nii.gz', 'JHU-tracts.xml'),
                         ('Juelich/Juelich-maxprob-thr0-1mm.nii.gz', 'Juelich.xml'),
                         ('MNI/MNI-maxprob-thr0-1mm.nii.gz', 'MNI.xml'),
                         ('SMATT/SMATT-labels-1mm.nii.gz', 'SMATT.xml'),
                         ('STN/STN-maxprob-thr0-0.5mm.nii.gz', 'STN.xml'),
                         ('Striatum/striatum-structural-1mm.nii.gz', 'Striatum-Structural.xml'),
                         ('Talairach/Talairach-labels-1mm.nii.gz', 'Talairach.xml'),
                         ('Thalamus/Thalamus-maxprob-thr0-1mm.nii.gz', 'Thalamus.xml')]

    _brain_directory = ""
    _save_location = ""
    _fsl_path = ""
    _atlas_path = ""
    _atlas_label_path = ""
    _atlas_name = ""
    _labelArray = []

    def __init__(self, brain, atlas="", atlas_path="", labels=""):
        self.brain = brain
        self.label_list = labels
        self.atlas_path = atlas_path

        self.no_ext_brain = atlas + "_" + os.path.splitext(self.brain)[0]
        self.tsnr_brain = self.stat_map_folder + os.path.splitext(self.brain)[0] + self.stat_map_suffix

        self.mean_brain = ""
        self.bet_brain = ""
        self.brain_to_mni = ""
        self.brain_to_mni_mat = ""
        self.invt_mni_transform = ""
        self.mni_to_brain = ""
        self.mni_to_brain_mat = ""
        self.roiTSNRs = ""
        self.roi_stat_list = ""
        self.file_list = []
        self.atlas_scale_filename = ['Voxels', 'Mean', 'Standard_Deviation',
                                     '%s_Confidence_Interval' % self._conf_level_list[int(self.conf_level_number)][0],
                                     'Min', 'Max']

    def __call__(self, brain_number_current, brain_number_total):
        if brain_number_current == 0:
            print('\n--- Analysis ---')

        print('\nAnalysing brain {brain_num_cur}/{brain_num_tot}: {brain}.\n'.format(brain_num_cur=brain_number_current+1,
                                                                                     brain_num_tot=brain_number_total,
                                                                                     brain=self.brain))
        self.roi_flirt_transform()
        self.roi_stats()

    def roi_flirt_transform(self):
        """Function which uses NiPype to transform the chosen atlas into the native space."""
        def mean_over_time():
            # If file is 4D, convert to 3D using "fslmaths -Tmean"
            tMean = fsl.MeanImage()
            tMean.inputs.in_file = self.brain
            self.mean_brain = tMean.inputs.out_file = self._save_location + self.no_ext_brain + '_mean.nii.gz'
            tMean.run()

            self.file_list.append(self.mean_brain)

        def brain_extraction():
            # Brain extraction
            bet = fsl.BET()
            bet.inputs.in_file = self.mean_brain
            bet.inputs.frac = self.frac_inten
            self.bet_brain = bet.inputs.out_file = self._save_location + self.no_ext_brain + '_bet.nii.gz'
            bet.run()

            self.file_list.append(self.bet_brain)

        def brain_to_mni():
            # Convert to MNI space
            flirt = fsl.FLIRT()
            flirt.inputs.in_file = self.brain
            flirt.inputs.reference = self._fsl_path + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
            flirt.inputs.dof = self.dof
            self.brain_to_mni = flirt.inputs.out_file = self._save_location + self.no_ext_brain + '_to_mni.nii.gz'
            self.brain_to_mni_mat = flirt.inputs.out_matrix_file = self._save_location + self.no_ext_brain + '_to_mni.mat'
            flirt.run()

            self.file_list.extend([self.brain_to_mni, self.brain_to_mni_mat])

        def invert_transform():
            invt = fsl.ConvertXFM()
            invt.inputs.in_file = self.brain_to_mni_mat
            invt.inputs.invert_xfm = True
            self.invt_mni_transform = invt.inputs.out_file = self._save_location + 'mni_to_' + self.no_ext_brain + '.mat'
            invt.run()

            self.file_list.append(self.invt_mni_transform)

        def mni_to_brain():
            flirt_inv = fsl.ApplyXFM()
            flirt_inv.inputs.in_file = self.atlas_path
            flirt_inv.inputs.reference = self.brain
            flirt_inv.inputs.in_matrix_file = self.invt_mni_transform
            flirt_inv.inputs.apply_xfm = True
            flirt_inv.inputs.interp = 'nearestneighbour'
            self.mni_to_brain = flirt_inv.inputs.out_file = self._save_location + 'mni_to_' + self.no_ext_brain + '.nii.gz'
            self.mni_to_brain_mat = flirt_inv.inputs.out_matrix_file = self._save_location + 'atlas_to_' + self.no_ext_brain + '.mat'
            flirt_inv.run()

            self.file_list.extend([self.mni_to_brain, self.mni_to_brain_mat])

        mean_over_time()

        if self.brain_extract:
            brain_extraction()

        brain_to_mni()
        invert_transform()
        mni_to_brain()

    def roi_stats(self):
        """Function which uses the output from the roi_flirt_transform function to collate the statistical information
        per ROI."""
        def calculate_stats():
            # Load original brain (with statistical map)
            tsnr_brain = nib.load(self.tsnr_brain)
            # Load atlas brain (which has been converted into native space)
            mni_brain = nib.load(self._save_location + 'mni_to_' + self.no_ext_brain + '.nii.gz')

            tsnr_brain = tsnr_brain.get_fdata()
            mni_brain = mni_brain.get_fdata()

            if mni_brain.shape != tsnr_brain.shape:
                raise Exception('The matrix dimensions of the standard space and the tSNR brain do not match.')

            # Find the number of unique ROIs in the atlas
            roiList = list(range(0, len(self.label_list) - 1))
            roiNum = np.size(roiList)

            idxBrain = tsnr_brain.flatten()
            idxMNI = mni_brain.flatten()

            # Create arrays to store the tSNR values before and after statistics
            roiTempStore = np.full([roiNum, idxMNI.shape[0]], np.nan)
            roiTSNRs = np.full([6, roiNum + 1], np.nan)

            # Where the magic happens
            for counter, roi in enumerate(idxMNI):
                roiTempStore[int(roi), counter] = idxBrain[counter]

            # Extract the roi parameters inputs and turn the necessary rows to nans to eliminate them from overall stat
            # calculations
            exclude_rois = None

            if self.include_rois != "all":
                include_rois = set(self.include_rois)  # Convert to set for performance
                exclude_rois = [number for number in roiList if number not in include_rois]
            elif self.exclude_rois != "none":
                exclude_rois = self.exclude_rois

            if exclude_rois:
                roiTempStore[exclude_rois, :] = np.nan

            warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

            roiTSNRs[0, 0:-1] = np.count_nonzero(~np.isnan(roiTempStore), axis=1) # Number of non-nan voxels in each ROI
            roiTSNRs[1, 0:-1] = np.nanmean(roiTempStore, axis=1)
            roiTSNRs[2, 0:-1] = np.nanstd(roiTempStore, axis=1)
            roiTSNRs[3, 0:-1] = self._conf_level_list[int(self.conf_level_number)][1] \
                                * roiTSNRs[2, 0:-1] / np.sqrt(roiTSNRs[0, 0:-1])  # 95% confidence interval calculation
            roiTSNRs[4, 0:-1] = np.nanmin(roiTempStore, axis=1)
            roiTSNRs[5, 0:-1] = np.nanmax(roiTempStore, axis=1)

            # Statistic calculations for all voxels assigned to an ROI
            if exclude_rois is None or 0 not in exclude_rois:
                start_val = 1
            else:
                start_val = 0

            roiTSNRs[0, -1] = np.count_nonzero(~np.isnan(roiTempStore[start_val:, :]))  # Number of non-nan voxels in total
            roiTSNRs[1, -1] = np.nanmean(roiTempStore[start_val:, :])
            roiTSNRs[2, -1] = np.nanstd(roiTempStore[start_val:, :])
            roiTSNRs[3, -1] = self._conf_level_list[int(self.conf_level_number)][1] \
                              * roiTSNRs[2, -1] / np.sqrt(roiTSNRs[0, -1])  # 95% CI calculation
            roiTSNRs[4, -1] = np.nanmin(roiTempStore[start_val:, :])
            roiTSNRs[5, -1] = np.nanmax(roiTempStore[start_val:, :])

            # Convert NaNs to zeros
            for column, voxel_num in enumerate(roiTSNRs[0]):
                if voxel_num == 0.0:
                    for row in list(range(1, 6)):
                        roiTSNRs[row][column] = 0.0

            warnings.filterwarnings('default')  # Reactivate warnings

            # Bootstrapping
            if self.bootstrap:
                for counter, roi in enumerate(list(range(0, roiNum-1))):
                    print(counter)  # TODO: indent this print statement and say it may take a while. "  - Bootstrapping roi 1/50."
                    roiTSNRs[3, roi] = Utils.calculate_confidence_interval(roiTempStore, roi) # TODO: Speed up code, multithreading?
                roiTSNRs[3, -1] = Utils.calculate_confidence_interval(roiTempStore, -1)  # TODO does this work? Not sure if -1 works

            headers = ['Voxels', 'Mean', 'Std_dev',
                       'Conf_Int_%s' % self._conf_level_list[int(self.conf_level_number)][0], 'Min', 'Max']

            # Save results as dataframe
            tsnr_result = pd.DataFrame(data=roiTSNRs,
                                       index=headers,
                                       columns=self.label_list)

            # Remove the required rows from the dataframe
            if exclude_rois:
                tsnr_result = tsnr_result.drop(tsnr_result.columns[exclude_rois], axis=1)

            # Save JSON file
            with open(self._save_location + self.no_ext_brain + ".json", 'w') as file:
                json.dump(tsnr_result.to_dict(), file, indent=2)

            # Save variable for atlas_scale function
            self.roiTSNRs = roiTSNRs

        def nipype_file_cleanup():
            """Clean up unnecessary output."""
            if self.save_stats_only:
                for file in self.file_list:
                    os.remove(file)
                return

        calculate_stats()
        nipype_file_cleanup()

    def atlas_scale(self, max_roi_stat, brain_number_current, brain_number_total):
        """Produces up to three scaled json files. Within brains, between brains (based on rois), between brains
        (based on the highest seen value of all brains and rois). Only the first json file will be created if using if
        running the analysis with all atlases."""
        if brain_number_current == 0:
            print('\n--- Image creation ---')

        print('\n Creating images for {brain}: {brain_num_cur}/{brain_num_tot}.\n'.format(brain_num_cur=brain_number_current+1,
                                                                                          brain_num_tot=brain_number_total,
                                                                                          brain=self.brain))

        within_brain_stat = nib.load(self.atlas_path)
        within_brain_stat = within_brain_stat.get_fdata()

        between_brain_stat = deepcopy(within_brain_stat)
        mixed_brain_stat = deepcopy(within_brain_stat)

        roi_scaled_stat = [(y/x) * 100 for x, y in zip(max_roi_stat, self.roiTSNRs[Analysis.roi_stat_number, :])]
        global_scaled_stat = [(y / max(max_roi_stat)) * 100 for y in self.roiTSNRs[Analysis.roi_stat_number, :]]

        roi_stat_brain_size = within_brain_stat.shape

        # Iterate through each voxel in the atlas
        for x in range(0, roi_stat_brain_size[0]):
            for y in range(0, roi_stat_brain_size[1]):
                for z in range(0, roi_stat_brain_size[2]):
                    # Set new value of voxel to the required statistic
                    roi_row = int(within_brain_stat[x][y][z])
                    if roi_row == 0:
                        within_brain_stat[x][y][z] = np.nan
                        between_brain_stat[x][y][z] = np.nan
                        mixed_brain_stat[x][y][z] = np.nan
                    else:
                        within_brain_stat[x][y][z] = self.roiTSNRs[self.roi_stat_number, roi_row]
                        between_brain_stat[x][y][z] = roi_scaled_stat[roi_row]
                        mixed_brain_stat[x][y][z] = global_scaled_stat[roi_row]

        # Convert atlas to NIFTI and save it
        affine = np.eye(4)
        scaled_atlas = nib.Nifti1Image(within_brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_atlas.nii.gz"
                                 % self.atlas_scale_filename[self.roi_stat_number])

        scaled_atlas = nib.Nifti1Image(between_brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_roi_scaled_atlas.nii.gz"
                                 % self.atlas_scale_filename[self.roi_stat_number])

        scaled_atlas = nib.Nifti1Image(mixed_brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_global_scaled_atlas.nii.gz"
                                 % self.atlas_scale_filename[self.roi_stat_number])

    @classmethod
    def batch_run_analysis(cls):
        try:
            cls._fsl_path = os.environ['FSLDIR']
        except OSError:
            raise Exception('FSL environment variable not set.')

        print('\n--- Environment Setup ---')

        if cls._brain_directory == "":
            print('Select the directory of the raw MRI/fMRI brains.')

            cls._brain_directory = Utils.file_browser()

            # Save copy of config.py to retain settings. It is saved here as after changing directory it will be harder
            # to find
            Utils.save_config(cls._brain_directory)

            os.chdir(cls._brain_directory)

        cls._atlas_name = os.path.splitext(Analysis._atlas_label_list[int(Analysis.atlas_number)][1])[0]
        print('Using the ' + cls._atlas_name + ' atlas.')

        cls._save_location = cls._atlas_name + "_ROI_report/"

        brain_file_nifti = [os.path.basename(f) for f in glob(cls._brain_directory + "/*.nii")]
        brain_file_hdr = [os.path.basename(f) for f in glob(cls._brain_directory + "/*.hdr")]
        cls.brain_file_list = brain_file_nifti + brain_file_hdr

        if len(cls.brain_file_list) == 0:
            raise NameError("No files found.")

        if not os.path.exists(cls._brain_directory + "/" + cls._save_location):
            os.mkdir(cls._brain_directory + "/" + cls._save_location)

        Utils.move_file('config.py', cls._brain_directory, cls._save_location)  # Move config file to analysis folder

        cls.roi_label_list()

        brain_class_list = []
        for brain in cls.brain_file_list:
            brain_class_list.append(Analysis(brain, atlas=cls._atlas_name, atlas_path=cls._atlas_path,
                                             labels=cls._labelArray))

        return brain_class_list

    @classmethod
    def roi_label_list(cls):
        cls._atlas_path = cls._fsl_path + '/data/atlases/' + cls._atlas_label_list[int(cls.atlas_number)][0]
        cls._atlas_label_path = cls._fsl_path + '/data/atlases/' + cls._atlas_label_list[int(cls.atlas_number)][1]

        with open(cls._atlas_label_path) as fd:
            atlas_label_dict = xmltodict.parse(fd.read())

        cls._labelArray = []
        cls._labelArray.append('No ROI')

        for roiLabelLine in atlas_label_dict['atlas']['data']['label']:
            cls._labelArray.append(roiLabelLine['#text'])

        cls._labelArray.append('Overall')

    @classmethod
    def help(cls):
        """Produce help text when the parameter "help" is passed with the filename. I.e. "roiAnalysis.py help"."""
        print("\n   Atlas list:")
        for counter, atlas in enumerate(cls._atlas_label_list):
            print("Atlas number " + str(counter) + ": " + os.path.splitext(atlas[1])[0])

        print("\n   Confidence level list:")
        for counter, level in enumerate(cls._conf_level_list):
            print("Confidence level number " + str(counter) + ": " + level[0] + "%")

        print("\n   Statistic list (to apply to ROIs in atlas):")
        for counter, stat in enumerate(cls._roi_stat_list):
            print(("Statistic number " + str(counter) + ": " + stat))

        print("\nEdit the config.py file to change tool parameters.")

        sys.exit()


class Figures:
    _brain_plot_file_extension = ["_Mean_atlas.nii.gz", "_Mean_roi_scaled_atlas.nii.gz",
                                  "_Mean_global_scaled_atlas.nii.gz"]
    # Config imported files
    dpi = config.plot_dpi
    font_size = config.plot_font_size
    scale = config.plot_scale

    make_figure_table = config.make_figure_table
    make_brain_table = config.make_brain_table
    make_one_region_fig = config.make_one_region_fig

    vmin = config.brain_fig_value_min
    vmax = config.brain_fig_value_max
    base_extension = _brain_plot_file_extension[config.brain_fig_file]
    brain_table_cols = config.brain_table_cols
    brain_table_rows = config.brain_table_rows
    brain_table_x_size = config.brain_table_x_size
    brain_table_y_size = config.brain_table_y_size
    brain_x_coord = config.brain_x_coord
    brain_z_coord = config.brain_z_coord

    table_cols = config.table_cols
    table_rows = config.table_rows
    ylab = config.table_y_label
    xlab = config.table_x_label

    single_fig_regions = config.single_roi_fig_regions
    single_fig_x_axis = config.single_roi_fig_x_axis
    single_fig_fill = config.single_roi_fig_colour
    single_fig_label_x = config.single_roi_fig_label_x
    single_fig_label_y = config.single_roi_fig_label_y
    single_fig_label_fill = config.single_roi_fig_label_fill
    colour_style = config.single_roi_fig_colours

    @classmethod
    def construct_plots(cls):
        print('\n--- Figure creation ---')

        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        plt.rcParams['figure.figsize'] = cls.brain_table_x_size, cls.brain_table_y_size  # Brain plot code

        df = pd.read_json("combined_results.json")

        if cls.make_one_region_fig:
            cls.one_region_plot(df)

        if cls.make_brain_table:
            cls.brain_plot(df)

        if cls.make_figure_table:
            cls.figure_table(df)

    @classmethod
    def figure_table(cls, df):
        df = df[(df['index'] != 'Overall') & (df['index'] != 'No ROI')]  # Remove No ROI and Overall rows

        # Group and sort df so figures are sorted according to the first panel
        df = df.groupby([cls.table_cols, cls.table_rows]).apply(lambda x: x.sort_values(['Mean']))  # Group by parameters and sort
        roi_ord = pd.Categorical(df['index'], categories=df['index'].unique())  # Set roi_ord to an ordered categorical
        df = df.assign(roi_cat=roi_ord)  # Create new column called roi_ord
        df = df.reset_index(drop=True)  # Reset index to remove grouping

        print("Preparing two parameter table!")
        figure_table = (pltn.ggplot(df, pltn.aes("Mean", "roi_ord"))
                        + pltn.geom_point(na_rm=True, size=1)
                        + pltn.geom_errorbarh(pltn.aes(xmin="Mean-Conf_Int_95", xmax="Mean+Conf_Int_95"),
                                              na_rm=True, height=None)
                        + pltn.scale_y_discrete(labels=[])
                        + pltn.ylab(cls.ylab)
                        + pltn.xlab(cls.xlab)
                        + pltn.facet_grid('{rows}~{cols}'.format(rows=cls.table_rows, cols=cls.table_cols),
                                          drop=True, labeller="label_both")
                        + pltn.theme_538()  # Set theme
                        + pltn.theme(panel_grid_major_y=pltn.themes.element_line(alpha=0),
                                     panel_grid_major_x=pltn.themes.element_line(alpha=1),
                                     panel_background=pltn.element_rect(fill="gray", alpha=0.1),
                                     dpi=cls.dpi))

        figure_table.save("Figures/two_param_table.png", height=cls.scale, width=cls.scale*3,
                          verbose=False, limitsize=False)
        print("Saved two parameter table!")

    @classmethod
    def one_region_plot(cls, df):
        list_rois = list(df['index'].unique())

        if cls.single_fig_regions is None:
            chosen_rois = []

            for roi_num, roi in enumerate(list_rois):
                print("{roi_num}: {roi}".format(roi_num=roi_num, roi=roi))
        else:
            if cls.single_fig_regions.lower() == "all":
                chosen_rois = list(range(0, len(list_rois)))
            else:
                chosen_rois = cls.single_fig_regions

        while not chosen_rois:
            roi_ans = input("Type a comma-separated list of the ROIs (listed above) you want to produce a figure for, "
                            "'e.g. 2,15,7,23' or 'all' for all rois. \nAlternatively press enter to skip this step: ")
            print("")

            if roi_ans.lower() == "all":
                chosen_rois = list(range(0, len(list_rois)))

            elif len(roi_ans) > 0:
                chosen_rois = roi_ans.split(",")  # Split by comma

                try:
                    chosen_rois = list(map(int, chosen_rois))  # Convert each list item to integers

                except ValueError:
                    print('Comma-separated list contains non integers.\n')
                    chosen_rois = []

            else:  # Else statement for blank input
                chosen_rois = []
                break

        for roi in chosen_rois:
            thisroi = list_rois[roi]

            current_df = df.loc[df['index'] == thisroi]

            current_df = current_df.sort_values([cls.single_fig_x_axis])
            current_df = current_df.reset_index(drop=True)  # Reset index to remove grouping
            current_df[cls.single_fig_x_axis] = pd.Categorical(current_df[cls.single_fig_x_axis],
                                                               categories=current_df[cls.single_fig_x_axis].unique())

            print("Setting up {thisroi}_barplot.png".format(thisroi=thisroi))

            figure = (pltn.ggplot(current_df, pltn.aes(x=cls.single_fig_x_axis, y='Mean',
                                                       ymin="Mean-Conf_Int_95", ymax="Mean+Conf_Int_95",
                                                       fill='factor({colour})'.format(colour=cls.single_fig_fill)))
                      + pltn.theme_538()
                      + pltn.geom_col(position=pltn.position_dodge(preserve='single', width=0.8), width=0.8, na_rm=True)
                      + pltn.geom_errorbar(size=1, position=pltn.position_dodge(preserve='single', width=0.8))
                      + pltn.labs(x=cls.single_fig_label_x, y=cls.single_fig_label_y, fill=cls.single_fig_label_fill)
                      + pltn.scale_x_discrete(labels=[])
                      + pltn.theme(panel_grid_major_x=pltn.element_line(alpha=0),
                                   axis_title_x=pltn.element_text(weight='bold', color='black', size=20),
                                   axis_title_y=pltn.element_text(weight='bold', color='black', size=20),
                                   axis_text_y=pltn.element_text(size=20, color='black'),
                                   legend_title=pltn.element_text(size=20, color='black'),
                                   legend_text=pltn.element_text(size=18, color='black'),
                                   subplots_adjust={'right': 0.85},
                                   legend_position=(0.9, 0.8),
                                   dpi=cls.dpi
                                   )
                      + pltn.geom_text(pltn.aes(y=-.7, label='MB'),
                                       color='black', size=20, va='top')
                      + pltn.scale_fill_manual(values=cls.colour_style)
                      # + pltn.scale_fill_grey()
                      )

            figure.save("Figures/{thisroi}_barplot.png".format(thisroi=thisroi), height=cls.scale, width=cls.scale*3,
                                                                                 verbose=False, limitsize=False)
            print("Saved {thisroi}_barplot.png".format(thisroi=thisroi))

    @classmethod
    def brain_plot(cls, df):
        print("Preparing brain table!")
        json_array = df['File_name'].unique()

        plot_values, axis_titles, current_params, col_nums, \
        row_nums, cell_nums, y_axis_size, x_axis_size = cls.table_setup(df)

        if cls.base_extension != "_Mean_atlas.nii.gz" and cls.vmax is None:
            cls.vmax = 100

        for file_num, json in enumerate(json_array):
            image_name = json + ".png"

            plt.subplot(y_axis_size, x_axis_size, cell_nums[file_num] + 1)
            plot = plotting.plot_anat(json + cls.base_extension,
                                      draw_cross=False, annotate=False, colorbar=True, display_mode='xz',
                                      vmin=cls.vmin, vmax=cls.vmax, cut_coords=(cls.brain_x_coord, cls.brain_z_coord),
                                      cmap='inferno')
            plot.savefig(image_name)
            plot.close()

            img = mpimg.imread(json + ".png")
            plt.imshow(img)

            ax = plt.gca()
            ax.set_yticks([])  # Remove y-axis ticks
            ax.axes.yaxis.set_ticklabels([])  # Remove y-axis labels

            ax.set_xticks([])  # Remove x-axis ticks
            ax.axes.xaxis.set_ticklabels([])  # Remove x-axis labels

            if row_nums[file_num] == 0:
                plt.title(axis_titles[0] + " " + plot_values[0][col_nums[file_num]], fontsize=cls.font_size)

            if col_nums[file_num] == 0:
                plt.ylabel(axis_titles[1] + " " + plot_values[1][row_nums[file_num]],
                           fontsize=cls.font_size)

        cls.label_blank_cell_axes(plot_values, axis_titles, cell_nums, x_axis_size, y_axis_size)

        plt.tight_layout()
        plt.savefig("Figures/brain_table.png", dpi=cls.dpi, bbox_inches='tight')
        print("Saved brain table!")

    @classmethod
    def table_setup(cls, df):
        unique_params = []
        current_params = []
        col_nums = []
        row_nums = []
        cell_nums = []

        for key in ParamParser.parameter_dict:
            params = sorted(list(df[key].unique()))  # Sort list of parameters
            params = [str(param) for param in params]  # Convert parameters to strings
            unique_params.append(params)

        plot_values = unique_params  # Get axis values
        axis_titles = list(ParamParser.parameter_dict.keys())  # Get axis titles
        plot_values_sorted = [plot_values[axis_titles.index(cls.brain_table_cols)],  # Sort axis values
                              plot_values[axis_titles.index(cls.brain_table_rows)]]

        x_axis_size = len(plot_values_sorted[0])
        y_axis_size = len(plot_values_sorted[1])

        for file_num, file_name in enumerate(df['File_name'].unique()):
            temp_param_store = []
            file_name_row = df[df['File_name'] == file_name].iloc[0]  # Get the first row of the relevant file name

            temp_param_store.append(str(file_name_row[cls.brain_table_cols]))
            temp_param_store.append(str(file_name_row[cls.brain_table_rows]))

            current_params.append(temp_param_store)  # Store parameters used for file

            col_nums.append(plot_values_sorted[0].index(current_params[file_num][0]))  # Work out col number
            row_nums.append(plot_values_sorted[1].index(current_params[file_num][1]))  # Work out row number

            cell_nums.append(np.ravel_multi_index((row_nums[file_num], col_nums[file_num]),
                                                  (y_axis_size, x_axis_size))) # Find linear index of figure

        return plot_values_sorted, axis_titles, current_params, col_nums, row_nums, cell_nums, y_axis_size, x_axis_size

    @classmethod
    def label_blank_cell_axes(cls, plot_values, axis_titles, cell_nums, x_axis_size, y_axis_size):
        for counter, x_title in enumerate(plot_values[0]):
            hidden_cell = np.ravel_multi_index((0, counter), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.title(axis_titles[0] + " " + x_title, fontsize=cls.font_size)

                if hidden_cell == 0:
                    plt.ylabel(axis_titles[1] + " " + plot_values[1][0], fontsize=cls.font_size)

                cls.make_cell_invisible()

        for counter, y_title in enumerate(plot_values[1]):
            hidden_cell = np.ravel_multi_index((counter, 0), (y_axis_size, x_axis_size))

            if hidden_cell not in cell_nums and hidden_cell != 0:
                plt.subplot(y_axis_size, x_axis_size, hidden_cell + 1)
                plt.ylabel(axis_titles[1] + " " + y_title, fontsize=cls.font_size)

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


class ParamParser:
    parameter_dict = config.parameter_dict
    binary_params = config.binary_params
    skip_verify_params = config.skip_verify_params

    @classmethod
    def run_parse(cls):
        parameter_values_raw = []
        combined_results_create = True

        json_array = cls.json_search()

        while True:
            if "combined_results.json" in json_array:
                answer = input("\nCombined results file already found in folder, "
                               "do you want to continue with this file or replace it? (continue or replace) ")

                if answer.lower() not in ('replace', 'continue'):
                    print("Not an appropriate choice.")
                else:
                    if answer.lower() == "replace":
                        os.remove("combined_results.json")
                        json_array.remove("combined_results.json")
                        print("File removed!")
                        combined_results_create = True
                    else:
                        combined_results_create = False
                    break
            else:
                break

        # Double check if the user wants to skip parameter verification
        if not ParamParser.skip_verify_params:
            skip_verify = input("Do you want to skip the verify parameter step? (y or n) ")
            skip_verify = skip_verify.lower()

            if skip_verify != "y":
                print("\nDo you want to verify the following MRI parameters during this step?")

                used_parameters = []
                for parameter in cls.parameter_dict:
                    answer = input(parameter + "? (y or n) ")

                    if answer.lower() in ("y", "yes"):
                        used_parameters.append(1)
                    else:
                        used_parameters.append(0)

        else:
            skip_verify = "y"

        combined_dataframe = pd.DataFrame()

        for json in json_array:
            param_nums = cls.parse_params(json)

            if skip_verify == "y":
                parameter_values_raw.append(param_nums)
            else:
                parameter_values_raw.append(cls.verify_params(json, param_nums, used_parameters))

            if combined_results_create:
                combined_dataframe = cls.construct_combined_json(combined_dataframe, json, param_nums)

        if combined_results_create:
            # Save combined results
            combined_dataframe = combined_dataframe.reset_index()
            combined_dataframe.to_json("combined_results.json", orient='records')

    @classmethod
    def parse_params(cls, json_file_name):
        """Search for MRI parameters in each json file name for use in table headers and created the combined json."""
        param_nums = []

        for key in cls.parameter_dict:
            parameter = cls.parameter_dict[key]  # Extract search term

            if parameter in cls.binary_params:
                param = re.search("{}".format(parameter), json_file_name, flags=re.IGNORECASE)

                if param is not None:
                    param_nums.append('On')  # Save 'on' if parameter is found in file name
                else:
                    param_nums.append('Off')  # Save 'off' if parameter not found in file name

            else:
                # Float search
                param = re.search("{}[0-9]p[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
                if param is not None:
                    param_nums.append(param[0][1] + "." + param[0][-1])
                    continue

                # If float search didnt work then Integer search
                param = re.search("{}[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
                if param is not None:
                    param_nums.append(param[0][-1])  # Extract the number from the parameter

                else:
                    param_nums.append(str(param))  # Save None if parameter not found in file name

        return param_nums

    @classmethod
    def verify_params(cls, json_file, param_nums, used_parameters):
        """Verify parsed parameters with user."""
        while True:
            print("\nFilename:\n" + json_file)

            print("\nParameters:")
            for counter, key in enumerate(cls.parameter_dict):
                if used_parameters[counter] == 1:
                    print("{} = {}".format(key, param_nums[counter]))

            while True:  # Ask user if the parsed parameters are correct
                answer = input("Is this correct? (y or n) ")

                if answer.lower() not in ('y', 'yes', 'n', 'no'):
                    print("Not an appropriate choice.")
                else:
                    break

            if answer.lower() == "y" or answer.lower() == "yes":
                return param_nums

            else:  # If parameters are not correct, ask for the actual values
                print("Please input the correct values. Non-numeric input will not change the original values.")
                for counter, key in enumerate(cls.parameter_dict):
                    if used_parameters[counter] == 1:
                        print("{} = {}".format(key, param_nums[counter]))
                        new_param_num = input("Actual value: ")

                        if re.match("[0-9]", new_param_num) and key not in cls.binary_params:
                            param_nums[counter] = new_param_num
                        elif key in cls.binary_params:
                            param_nums[counter] = new_param_num

                print("\nRe-checking...")  # Repeat the verification process

    @staticmethod
    def json_search():
        print('Select the directory of json files.')

        json_directory = Utils.file_browser()

        os.chdir(json_directory)

        json_file_list = [os.path.basename(f) for f in glob(json_directory + "/*.json")]

        if len(json_file_list) == 0:
            raise NameError('No json files found.')
        else:
            return json_file_list

    @classmethod
    def construct_combined_json(cls, dataframe, json, parameters):
        if dataframe.empty:
            dataframe = pd.read_json(json)
            dataframe = dataframe.transpose()

            for counter, parameter_name in enumerate(cls.parameter_dict):
                dataframe[parameter_name] = parameters[counter]
                dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename
        else:
            new_dataframe = pd.read_json(json)
            new_dataframe = new_dataframe.transpose()
            for counter, parameter_name in enumerate(cls.parameter_dict):
                new_dataframe[parameter_name] = parameters[counter]  # Add parameter columns

            new_dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename

            dataframe = dataframe.append(new_dataframe, sort=True)

        return dataframe


class Utils:
    @staticmethod
    def file_browser():
        root = Tk()  # Create tkinter window

        root.withdraw()  # Hide tkinter window
        root.update()

        directory = filedialog.askdirectory()

        root.update()
        root.destroy()  # Destroy tkinter window

        return directory

    @staticmethod
    def save_config(directory):
        with open(directory + '/config.py', 'w') as f:
            with open('config.py', 'r') as r:
                for line in r:
                    f.write(line)

    @staticmethod
    def calculate_confidence_interval(data, roi):
        values = csr_matrix([x for x in data[roi, :] if str(x) != 'nan'])  # TODO
        results = bs.bootstrap(values, stat_func=bs_stats.mean)
        conf_int = results.value - results.lower_bound

        return conf_int

    @staticmethod
    def move_file(name, original_dir, new_dir):
        os.rename(original_dir + '/' + name, new_dir + '/' + name)


if __name__ == '__main__':
    # Run help if passed as parameter
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            Analysis.help()

    # Running the full analysis is optional
    if Analysis.run_analysis:
        # Run class setup
        brain_list = Analysis.batch_run_analysis()

        # Run analysis
        for brain_counter, brain in enumerate(brain_list):
            brain(brain_counter, len(brain_list))

        # Atlas scaling
        '''Save a copy of the stats (default mean) for each ROI from the first brain. Then using sequential comparison
        find the largest ROI stat out of all the brains analyzed.'''
        roi_stats = deepcopy(brain_list[0].roiTSNRs[Analysis.roi_stat_number, :])
        for brain in brain_list:
            for counter, roi_stat in enumerate(brain.roiTSNRs[Analysis.roi_stat_number, :]):
                if roi_stat > roi_stats[counter]:
                    roi_stats[counter] = roi_stat

        # Run atlas_scale function and pass in max roi stats for between brain scaling
        for brain_counter, brain in enumerate(brain_list):
            brain.atlas_scale(roi_stats, brain_counter, len(brain_list))

    # Parameter Parsing
    ParamParser.run_parse()

    # Plotting
    Figures.construct_plots()

    print('Done!')




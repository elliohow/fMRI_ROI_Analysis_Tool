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
import concurrent.futures  # TODO: Implement

from copy import deepcopy
from scipy.sparse import csr_matrix
from glob import glob
from nipype.interfaces import fsl, freesurfer
from matplotlib import pyplot as plt
from nilearn import plotting
from tkinter import Tk, filedialog


class Analysis:
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
        self.stat_brain = config.stat_map_folder + os.path.splitext(self.brain)[0] + config.stat_map_suffix

        self.mean_brain = ""
        self.bet_brain = ""
        self.brain_to_mni = ""
        self.brain_to_mni_mat = ""
        self.invt_mni_transform = ""
        self.mni_to_brain = ""
        self.mni_to_brain_mat = ""
        self.roiResults = ""
        self.roi_stat_list = ""
        self.file_list = []
        self.atlas_scale_filename = ['Voxels', 'Mean', 'Standard_Deviation',
                                     '%s_Confidence_Interval' % self._conf_level_list[int(config.conf_level_number)][0],
                                     'Min', 'Max']

    def __call__(self, brain_number_current, brain_number_total, freesurfer_excluded_voxels):
        if brain_number_current == 0:
            print('\n--- Analysis ---')

        print('\nAnalysing brain {brain_num_cur}/{brain_num_tot}: {brain}.\n'.format(
            brain_num_cur=brain_number_current + 1,
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
            bet.inputs.frac = config.frac_inten
            self.bet_brain = bet.inputs.out_file = self._save_location + self.no_ext_brain + '_bet.nii.gz'
            bet.run()

            self.file_list.append(self.bet_brain)

        def brain_to_mni():
            # Convert to MNI space
            flirt = fsl.FLIRT()
            flirt.inputs.in_file = self.bet_brain
            flirt.inputs.reference = self._fsl_path + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
            flirt.inputs.dof = config.dof
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
            flirt_inv.inputs.reference = self.bet_brain
            flirt_inv.inputs.in_matrix_file = self.invt_mni_transform
            flirt_inv.inputs.apply_xfm = True
            flirt_inv.inputs.interp = 'nearestneighbour'
            self.mni_to_brain = flirt_inv.inputs.out_file = self._save_location + 'mni_to_' + self.no_ext_brain + '.nii.gz'
            self.mni_to_brain_mat = flirt_inv.inputs.out_matrix_file = self._save_location + 'atlas_to_' + self.no_ext_brain + '.mat'
            flirt_inv.run()

            self.file_list.extend([self.mni_to_brain, self.mni_to_brain_mat])

        mean_over_time()
        brain_extraction()
        brain_to_mni()
        invert_transform()
        mni_to_brain()

    def roi_stats(self, excluded_voxels=None):
        """Function which uses the output from the roi_flirt_transform function to collate the statistical information
        per ROI."""

        def calculate_stats():
            # Load original brain (with statistical map)
            stat_brain = nib.load(self.stat_brain)
            # Load atlas brain (which has been converted into native space)
            mni_brain = nib.load(self._save_location + 'mni_to_' + self.no_ext_brain + '.nii.gz')

            stat_brain = stat_brain.get_fdata()
            mni_brain = mni_brain.get_fdata()

            if mni_brain.shape != stat_brain.shape:
                raise Exception('The matrix dimensions of the standard space and the statistical map brain do not '
                                'match.')

            # Find the number of unique ROIs in the atlas
            roiList = list(range(0, len(self.label_list) - 1))
            roiNum = np.size(roiList)

            idxBrain = stat_brain.flatten()
            idxMNI = mni_brain.flatten()

            # Create arrays to store the values before and after statistics
            roiTempStore = np.full([roiNum, idxMNI.shape[0]], np.nan)
            roiResults = np.full([7, roiNum + 1], np.nan)
            roiResults[6, 0:-1] = 0  # Change freesurfer exclude voxels measure from NaN to 0

            # Where the magic happens
            if config.use_freesurf_file is not False:
                for counter, roi in enumerate(idxMNI):
                    if excluded_voxels[counter] == 0:
                        roiTempStore[int(roi), counter] = idxBrain[counter]
                    else:
                        roiResults[6, int(roi)] += 1
            else:
                for counter, roi in enumerate(idxMNI):
                    roiTempStore[int(roi), counter] = idxBrain[counter]

            # Extract the roi parameters inputs and turn the necessary rows to nans to eliminate them from overall stat
            # calculations
            exclude_rois = None

            if config.include_rois != "all":
                include_rois = set(config.include_rois)  # Convert to set for performance
                exclude_rois = [number for number in roiList if number not in include_rois]
            elif config.exclude_rois != "none":
                exclude_rois = config.exclude_rois

            if exclude_rois:
                roiTempStore[exclude_rois, :] = np.nan

            warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

            roiResults[0, 0:-1] = np.count_nonzero(~np.isnan(roiTempStore),
                                                   axis=1)  # Number of non-nan voxels in each ROI
            roiResults[1, 0:-1] = np.nanmean(roiTempStore, axis=1)
            roiResults[2, 0:-1] = np.nanstd(roiTempStore, axis=1)
            roiResults[3, 0:-1] = self._conf_level_list[int(config.conf_level_number)][1] \
                                  * roiResults[2, 0:-1] / np.sqrt(
                roiResults[0, 0:-1])  # 95% confidence interval calculation
            roiResults[4, 0:-1] = np.nanmin(roiTempStore, axis=1)
            roiResults[5, 0:-1] = np.nanmax(roiTempStore, axis=1)

            # Statistic calculations for all voxels assigned to an ROI
            # If No ROI is part of excluded ROIs, start overall ROI calculation from 0, otherwise start from 1
            if exclude_rois is None or 0 not in exclude_rois:
                start_val = 1
            else:
                start_val = 0

            roiResults[0, -1] = np.count_nonzero(
                ~np.isnan(roiTempStore[start_val:, :]))  # Number of non-nan voxels in total
            roiResults[1, -1] = np.nanmean(roiTempStore[start_val:, :])
            roiResults[2, -1] = np.nanstd(roiTempStore[start_val:, :])
            roiResults[3, -1] = self._conf_level_list[int(config.conf_level_number)][1] \
                                * roiResults[2, -1] / np.sqrt(roiResults[0, -1])  # 95% CI calculation
            roiResults[4, -1] = np.nanmin(roiTempStore[start_val:, :])
            roiResults[5, -1] = np.nanmax(roiTempStore[start_val:, :])
            roiResults[6, -1] = np.sum(roiResults[6, start_val:-2])

            # Convert NaNs to zeros
            for column, voxel_num in enumerate(roiResults[0]):
                if voxel_num == 0.0:
                    for row in list(range(1, 6)):
                        roiResults[row][column] = 0.0

            warnings.filterwarnings('default')  # Reactivate warnings

            # Bootstrapping
            if config.bootstrap:
                for counter, roi in enumerate(list(range(0, roiNum))):
                    if counter == 0:
                        print("This may take a while...")
                    print("  - Bootstrapping roi {}/{}".format(counter, roiNum))
                    roiResults[3, roi] = Utils.calculate_confidence_interval(roiTempStore,
                                                                             roi=roi)  # TODO: Speed up code, multithreading?
                # Calculate overall statistics
                roiResults[3, -1] = Utils.calculate_confidence_interval(
                    roiTempStore[start_val:, :])  # TODO does this work?

            headers = ['Voxels', 'Mean', 'Std_dev',
                       'Conf_Int_%s' % self._conf_level_list[int(config.conf_level_number)][0],
                       'Min', 'Max', 'Freesurfer excluded voxels']

            # Save results as dataframe
            results = pd.DataFrame(data=roiResults,
                                   index=headers,
                                   columns=self.label_list)

            # Remove the required rows from the dataframe
            if exclude_rois:
                results = results.drop(results.columns[exclude_rois], axis=1)

            # Save JSON file
            with open(self._save_location + self.no_ext_brain + ".json", 'w') as file:
                json.dump(results.to_dict(), file, indent=2)

            # Save variable for atlas_scale function
            self.roiResults = roiResults

        def nipype_file_cleanup():
            """Clean up unnecessary output."""
            if config.save_stats_only:
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

        print('\n Creating images for {brain}: {brain_num_cur}/{brain_num_tot}.\n'.format(
            brain_num_cur=brain_number_current + 1,
            brain_num_tot=brain_number_total,
            brain=self.brain))

        within_brain_stat = nib.load(self.atlas_path)
        within_brain_stat = within_brain_stat.get_fdata()

        between_brain_stat = deepcopy(within_brain_stat)
        mixed_brain_stat = deepcopy(within_brain_stat)

        roi_scaled_stat = [(y / x) * 100 for x, y in zip(max_roi_stat, self.roiResults[config.roi_stat_number, :])]
        global_scaled_stat = [(y / max(max_roi_stat)) * 100 for y in self.roiResults[config.roi_stat_number, :]]

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
                        within_brain_stat[x][y][z] = self.roiResults[config.roi_stat_number, roi_row]
                        between_brain_stat[x][y][z] = roi_scaled_stat[roi_row]
                        mixed_brain_stat[x][y][z] = global_scaled_stat[roi_row]

        # Convert atlas to NIFTI and save it
        affine = np.eye(4)
        scaled_atlas = nib.Nifti1Image(within_brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_atlas.nii.gz"
                                 % self.atlas_scale_filename[config.roi_stat_number])

        scaled_atlas = nib.Nifti1Image(between_brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_roi_scaled_atlas.nii.gz"
                                 % self.atlas_scale_filename[config.roi_stat_number])

        scaled_atlas = nib.Nifti1Image(mixed_brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_global_scaled_atlas.nii.gz"
                                 % self.atlas_scale_filename[config.roi_stat_number])

    @staticmethod
    def freesurfer_space_to_native_space():
        """Function which removes freesurfer padding and transforms freesurfer segmentation to native space."""
        native_segmented_brain = freesurfer.Label2Vol(seg_file='freesurfer/aparc+aseg.mgz',
                                                      template_file='freesurfer/rawavg.mgz',
                                                      vol_label_file='freesurfer/native_segmented_brain.mgz',
                                                      reg_header='freesurfer/aparc+aseg.mgz')
        native_segmented_brain.run()

        freesurf_native_segmented_brain = nib.load('freesurfer/native_segmented_brain.mgz')
        freesurf_native_segmented_brain = freesurf_native_segmented_brain.get_fdata().flatten()
        # TODO: is this method chaining valid? freesurf_native_segmented_brain = freesurf_native_segmented_brain.flatten()

        # Make option of restricting to grey matter only

        # Using set instead of list for performance reasons
        # Refers to freesurfer lookup table of CSF and white matter
        csf_wm_values = {0, 2, 4, 5, 7, 14, 15, 24, 25, 41, 43, 44, 46, 72, 77, 78, 79, 98, 159, 160, 161, 162,
                         165, 167, 168, 177, 213, 219, 221, 223, 498, 499, 690, 691, 701, 703, 3000, 3001,
                         3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015,
                         3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029,
                         3030, 3031, 3032, 3033, 3034, 3035, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007,
                         4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021,
                         4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035,
                         3201, 3203, 3204, 3205, 3206, 3207, 4201, 4203, 4204, 4205, 4206, 4207, 3100, 3101,
                         3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115,
                         3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129,
                         3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143,
                         3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157,
                         3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171,
                         3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 4100, 4101, 4102, 4103,
                         4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117,
                         4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131,
                         4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145,
                         4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159,
                         4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173,
                         4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 5001, 5002, 13100, 13101, 13102,
                         13103, 13104, 13105, 13106, 13107, 13108, 13109, 13110, 13111, 13112, 13113, 13114,
                         13115, 13116, 13117, 13118, 13119, 13120, 13121, 13122, 13123, 13124, 13125, 13126,
                         13127, 13128, 13129, 13130, 13131, 13132, 13133, 13134, 13135, 13136, 13137, 13138,
                         13139, 13140, 13141, 13142, 13143, 13144, 13145, 13146, 13147, 13148, 13149, 13150,
                         13151, 13152, 13153, 13154, 13155, 13156, 13157, 13158, 13159, 13160, 13161, 13162,
                         13163, 13164, 13165, 13166, 13167, 13168, 13169, 13170, 13171, 13172, 13173, 13174,
                         13175, 14100, 14101, 14102, 14103, 14104, 14105, 14106, 14107, 14108, 14109, 14110,
                         14111, 14112, 14113, 14114, 14115, 14116, 14117, 14118, 14119, 14120, 14121, 14122,
                         14123, 14124, 14125, 14126, 14127, 14128, 14129, 14130, 14131, 14132, 14133, 14134,
                         14135, 14136, 14137, 14138, 14139, 14140, 14141, 14142, 14143, 14144, 14145, 14146,
                         14147, 14148, 14149, 14150, 14151, 14152, 14153, 14154, 14155, 14156, 14157, 14158,
                         14159, 14160, 14161, 14162, 14163, 14164, 14165, 14166, 14167, 14168, 14169, 14170,
                         14171, 14172, 14173, 14174, 14175}

        idxCSF_or_WM = np.full([freesurf_native_segmented_brain.shape], 0)

        # Get list of voxels, made duplicate of 1's, if voxel is of a value in the list above then set to 0.
        for counter, voxel in enumerate(freesurf_native_segmented_brain):
            if voxel in csf_wm_values:
                idxCSF_or_WM[counter] = 1

        return idxCSF_or_WM

    @classmethod
    def roi_label_list(cls):
        """Extract labels from specified FSL atlas XML file."""
        cls._atlas_path = cls._fsl_path + '/data/atlases/' + cls._atlas_label_list[int(config.atlas_number)][0]
        cls._atlas_label_path = cls._fsl_path + '/data/atlases/' + cls._atlas_label_list[int(config.atlas_number)][1]

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
                                  "_Mean_global_scaled_atlas.nii.gz", "all"]
    base_extension = _brain_plot_file_extension[config.brain_fig_file]

    @classmethod
    def construct_plots(cls):
        print('\n--- Figure creation ---')

        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        plt.rcParams['figure.figsize'] = config.brain_table_x_size, config.brain_table_y_size  # Brain plot code

        df = pd.read_json("combined_results.json")

        if config.make_one_region_fig:
            cls.one_region_bar_chart(df)

        if config.make_brain_table:
            if Figures.base_extension == "all":
                for base_extension in Figures._brain_plot_file_extension[0:-1]:
                    cls.brain_facet_grid(df, base_extension)
            else:
                cls.brain_facet_grid(df, Figures.base_extension)

        if config.make_figure_table:
            cls.scatter_plot(df)

    @classmethod
    def scatter_plot(cls, df):
        df = df[(df['index'] != 'Overall') & (df['index'] != 'No ROI')]  # Remove No ROI and Overall rows

        # Group and sort df so figures are sorted according to the first panel
        df = df.groupby([config.table_cols, config.table_rows]).apply(
            lambda x: x.sort_values(['Mean']))  # Group by parameters and sort
        roi_ord = pd.Categorical(df['index'], categories=df['index'].unique())  # Set roi_ord to an ordered categorical
        df = df.assign(roi_cat=roi_ord)  # Create new column called roi_ord
        df = df.reset_index(drop=True)  # Reset index to remove grouping

        print("Preparing two parameter table!")
        figure_table = (pltn.ggplot(df, pltn.aes("Mean", "roi_ord"))
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

        figure_table.save("Figures/two_param_table.png", height=config.plot_scale, width=config.plot_scale * 3,
                          verbose=False, limitsize=False)
        print("Saved two parameter table!")

    @classmethod
    def one_region_bar_chart(cls, df):
        list_rois = list(df['index'].unique())

        if config.single_roi_fig_regions is None:
            chosen_rois = []

            for roi_num, roi in enumerate(list_rois):
                print("{roi_num}: {roi}".format(roi_num=roi_num, roi=roi))
        else:
            if config.single_roi_fig_regions.lower() == "all":
                chosen_rois = list(range(0, len(list_rois)))
            else:
                chosen_rois = config.single_roi_fig_regions

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

            current_df = current_df.sort_values([config.single_roi_fig_x_axis])
            current_df = current_df.reset_index(drop=True)  # Reset index to remove grouping
            current_df[config.single_roi_fig_x_axis] = pd.Categorical(current_df[config.single_roi_fig_x_axis],
                                                                      categories=current_df[
                                                                          config.single_roi_fig_x_axis].unique())

            print("Setting up {thisroi}_barplot.png".format(thisroi=thisroi))

            figure = (pltn.ggplot(current_df, pltn.aes(x=config.single_roi_fig_x_axis, y='Mean',
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
                      + pltn.geom_text(pltn.aes(y=-.7, label='MB'),
                                       color='black', size=20, va='top')
                      + pltn.scale_fill_manual(values=config.single_roi_fig_colours)
                      # + pltn.scale_fill_grey()
                      )

            figure.save("Figures/{thisroi}_barplot.png".format(thisroi=thisroi), height=config.plot_scale,
                        width=config.plot_scale * 3,
                        verbose=False, limitsize=False)
            print("Saved {thisroi}_barplot.png".format(thisroi=thisroi))

    @classmethod
    def brain_facet_grid(cls, df, base_extension):
        base_ext_clean = os.path.splitext(os.path.splitext(base_extension)[0])[0][1:]
        print(f"Preparing {base_ext_clean} table!")
        json_array = df['File_name'].unique()

        plot_values, axis_titles, current_params, col_nums, \
        row_nums, cell_nums, y_axis_size, x_axis_size = cls.table_setup(df)

        if base_extension != "_Mean_atlas.nii.gz" and config.brain_fig_value_max is None:
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

        plt.tight_layout()
        plt.savefig(f"Figures/{base_ext_clean}.png", dpi=config.plot_dpi, bbox_inches='tight')
        plt.close()
        print("Saved brain table!")

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


class ParamParser:
    @classmethod
    def run_parse(cls):
        combined_results_create = True

        json_array = cls.json_search()

        while True and not config.always_replace_combined_json:
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
        if config.verify_param_method == "manual":
            skip_verify = input(
                "Do you want to skip the verification of parameters and rely on file names instead? (y or n) ")
            skip_verify = skip_verify.lower()

            if skip_verify != "y":
                print("\nDo you want to verify the following MRI parameters during this step?")

                used_parameters = []
                for parameter in config.parameter_dict:
                    answer = input(parameter + "? (y or n) ")

                    if answer.lower() in ("y", "yes"):
                        used_parameters.append(1)
                    else:
                        used_parameters.append(0)

        else:
            skip_verify = "y"

        combined_dataframe = pd.DataFrame()

        if config.verify_param_method == "table":
            table = pd.read_csv(config.param_table_name)  # Load param table

        for json in json_array:
            if json == "combined_results.json":
                continue

            if config.verify_param_method in ("manual", "name"):
                param_nums = cls.parse_params_from_file_name(json)
            else:
                param_nums = cls.parse_params_from_table_file(json, table)

            if skip_verify != "y":
                param_nums = cls.manually_verify_params(json, param_nums, used_parameters)

            if combined_results_create and param_nums:
                combined_dataframe = cls.construct_combined_json(combined_dataframe, json, param_nums)

        if combined_results_create:
            # Save combined results
            combined_dataframe = combined_dataframe.reset_index()
            combined_dataframe.to_json("combined_results.json", orient='records')

    @classmethod
    def parse_params_from_table_file(cls, json_file_name, table):
        # Find atlas used to remove text from the start of the json file name
        with open('config_log.py', 'r') as config_log:
            for line in config_log:
                line = line.rstrip()  # remove '\n' at end of line

                atlas_number = re.match("atlas_number = [0-9]", line)  # Search for atlas used from config_log
                if atlas_number:
                    atlas_name = os.path.splitext(Analysis._atlas_label_list[int(atlas_number[0][-1])][1])[0] + "_"

                    break

        json_file_name = json_file_name[len(atlas_name):]  # Remove atlas name prefix from file name
        json_file_name = os.path.splitext(json_file_name)[0]  # Remove file extension

        table_row = table.loc[table["File name"] == json_file_name]

        param_nums = []
        if table_row['Ignore file? (y for yes, otherwise blank)'].to_string(index=False).strip().lower() == 'y':
            return param_nums
        else:
            for key in config.parameter_dict:
                param_nums.append(float(table_row[key]))

        return param_nums

    @classmethod
    def parse_params_from_file_name(cls, json_file_name):
        """Search for MRI parameters in each json file name for use in table headers and created the combined json."""
        param_nums = []

        for key in config.parameter_dict:
            parameter = config.parameter_dict[key]  # Extract search term

            if parameter in config.binary_params:
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
    def manually_verify_params(cls, json_file, param_nums, used_parameters):
        """Verify parsed parameters with user."""
        while True:
            print("\nFilename:\n" + json_file)

            print("\nParameters:")
            for counter, key in enumerate(config.parameter_dict):
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
                for counter, key in enumerate(config.parameter_dict):
                    if used_parameters[counter] == 1:
                        print("{} = {}".format(key, param_nums[counter]))
                        new_param_num = input("Actual value: ")

                        if re.match("[0-9]", new_param_num) and key not in config.binary_params:
                            param_nums[counter] = new_param_num
                        elif key in config.binary_params:
                            param_nums[counter] = new_param_num

                print("\nRe-checking...")  # Repeat the verification process

    @staticmethod
    def json_search():
        if config.run_steps == "all":
            json_directory = os.getcwd() + f"/{Analysis._save_location}"

            if config.verify_param_method == "table":  # Move excel file containing parameter file info
                Utils.move_file(config.param_table_name, os.getcwd(), json_directory)

        else:
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

            for counter, parameter_name in enumerate(config.parameter_dict):
                dataframe[parameter_name] = parameters[counter]
                dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename
        else:
            new_dataframe = pd.read_json(json)
            new_dataframe = new_dataframe.transpose()

            for counter, parameter_name in enumerate(config.parameter_dict):
                new_dataframe[parameter_name] = parameters[counter]  # Add parameter columns

            new_dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename

            dataframe = dataframe.append(new_dataframe, sort=True)

        return dataframe


class Utils:
    @staticmethod
    def find_brain_files(brain_directory):
        brain_file_nifti = [os.path.basename(f) for f in glob(brain_directory + "/*.nii")]
        brain_file_hdr = [os.path.basename(f) for f in glob(brain_directory + "/*.hdr")]

        return brain_file_nifti + brain_file_hdr

    @staticmethod
    def file_browser(chdir=False):
        root = Tk()  # Create tkinter window

        root.withdraw()  # Hide tkinter window
        root.update()

        directory = filedialog.askdirectory()

        root.update()
        root.destroy()  # Destroy tkinter window

        if chdir:
            os.chdir(directory)

        return directory

    @staticmethod
    def save_config(directory):
        with open(directory + '/config_log.py', 'w') as f:
            with open('config.py', 'r') as r:
                for line in r:
                    f.write(line)

    @staticmethod
    def calculate_confidence_interval(data, roi=None):
        if roi is not None:
            values = csr_matrix([x for x in data[roi, :] if str(x) != 'nan'])
        else:
            data = data.flatten()
            values = csr_matrix([x for x in data if str(x) != 'nan'])
        results = bs.bootstrap(values, stat_func=bs_stats.mean)  # TODO how does this work with excluded voxels
        conf_int = results.value - results.lower_bound  # TODO what does this return

        return conf_int

    @staticmethod
    def move_file(name, original_dir, new_dir):
        os.rename(original_dir + '/' + name, new_dir + '/' + name)

    @classmethod
    def setup_analysis(cls):
        """Set up environment and find files before running analysis."""
        try:
            Analysis._fsl_path = os.environ['FSLDIR']
        except OSError:
            raise Exception('FSL environment variable not set.')

        print('\n--- Environment Setup ---')

        if Analysis._brain_directory == "":
            print('Select the directory of the raw MRI/fMRI brains.')

            Analysis._brain_directory = Utils.file_browser(chdir=True)

            # Save copy of config.py to retain settings. It is saved here as after changing directory it will be harder
            # to find
            cls.save_config(Analysis._brain_directory)

        Analysis._atlas_name = os.path.splitext(Analysis._atlas_label_list[int(config.atlas_number)][1])[0]
        print('Using the ' + Analysis._atlas_name + ' atlas.')

        Analysis_save_location = Analysis._atlas_name + "_ROI_report/"

        # Find all nifti and analyze files
        Analysis.brain_file_list = cls.find_brain_files(Analysis._brain_directory)

        if len(Analysis.brain_file_list) == 0:
            raise NameError("No files found.")

        # Make folder to save ROI_report if not already created
        if not os.path.exists(Analysis._brain_directory + "/" + Analysis._save_location):
            os.mkdir(Analysis._brain_directory + "/" + Analysis._save_location)

        cls.move_file('config_log.py', Analysis._brain_directory,
                      Analysis._save_location)  # Move config file to analysis folder

        # Extract labels from selected FSL atlas
        Analysis.roi_label_list()

        brain_class_list = []
        for brain in Analysis.brain_file_list:
            # Initialise Analysis class for each file found
            brain_class_list.append(Analysis(brain, atlas=Analysis._atlas_name, atlas_path=Analysis._atlas_path,
                                             labels=Analysis._labelArray))

        return brain_class_list

    @classmethod
    def make_table(cls):
        print('Select the nifti/analyse file directory.')
        brain_directory = cls.file_browser(chdir=True)

        brain_file_list = cls.find_brain_files(brain_directory)
        brain_file_list = [os.path.splitext(brain)[0] for brain in brain_file_list]
        brain_file_list.sort()

        padding = np.empty((len(brain_file_list)))
        padding[:] = np.NaN

        params = []
        for file in brain_file_list:
            # Try to find parameters to prefill table
            params.append(ParamParser.parse_params_from_file_name(file))
        params = np.array(params).transpose()

        df = pd.DataFrame(data={'File name': brain_file_list})

        for counter, param in enumerate(config.parameter_dict):
            df[param] = params[counter]

        df['Ignore file? (y for yes, otherwise blank)'] = padding

        df.to_csv('paramValues.csv', index=False)

        print(f"\nparamValues.csv saved in {brain_directory}.\n\nInput parameter values in paramValues.csv and change "
              f"make_table_only to False in the config file to continue analysis. \nIf analysis has already been "
              f"conducted, move paramValues.csv into the ROI report folder. \nIf the csv file contains unexpected "
              f"parameters, update config.parameter_dict.")

        sys.exit()


if __name__ == '__main__':
    # Run help if passed as parameter
    if len(sys.argv) > 1:
        if sys.argv[1] == 'help':
            Analysis.help()

    if config.make_table_only:
        Utils.make_table()

    # Running the full analysis is optional
    if config.run_steps in ("analyse", "all"):
        # Run class setup
        brain_list = Utils.setup_analysis()

        if Analysis.use_freesurf_file:
            csf_or_wm_voxels = Analysis.freesurfer_space_to_native_space()
        else:
            csf_or_wm_voxels = None

        # Run analysis
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # TODO: check this parallel processing works (Note: it doesnt currently)
            # for brain_counter, brain in enumerate(brain_list):
            #     executor.submit(brain(brain_counter, len(brain_list), freesurfer_excluded_voxels=csf_or_wm_voxels))

            futures = {executor.submit(brain, counter, len(brain_list), freesurfer_excluded_voxels=csf_or_wm_voxels):
                           (counter, brain) for counter, brain in enumerate(brain_list)}

            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        # Atlas scaling
        '''Save a copy of the stats (default mean) for each ROI from the first brain. Then using sequential comparison
        find the largest ROI stat out of all the brains analyzed.'''
        roi_stats = deepcopy(brain_list[0].roiResults[config.roi_stat_number, :])
        for brain in brain_list:
            for counter, roi_stat in enumerate(brain.roiResults[config.roi_stat_number, :]):
                if roi_stat > roi_stats[counter]:
                    roi_stats[counter] = roi_stat

        # Run atlas_scale function and pass in max roi stats for between brain scaling
        for brain_counter, brain in enumerate(brain_list):
            brain.atlas_scale(roi_stats, brain_counter, len(brain_list))

    if config.run_steps in ("plot", "all"):
        # Parameter Parsing
        ParamParser.run_parse()

        # Plotting
        Figures.construct_plots()

    print('Done!')

import os
import sys
import warnings
from copy import deepcopy

import nibabel as nib
import numpy as np
import pandas as pd
import simplejson as json
import xmltodict
from nipype.interfaces import fsl, freesurfer

import config
from roianalysis.utils import Utils


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

        self.mc_brain = ""
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

        # Copying class attributes here is a workaround for dill, which can't access modified class attributes for
        # imported modules.
        self._brain_directory = self._brain_directory
        self._save_location = self._save_location
        self._fsl_path = self._fsl_path
        self._atlas_path = self._atlas_path
        self._atlas_label_path = self._atlas_label_path
        self._atlas_name = self._atlas_name
        self._labelArray = self._labelArray

    @staticmethod
    def setup_analysis():
        """Set up environment and find files before running analysis."""
        try:
            Analysis._fsl_path = os.environ['FSLDIR']
        except OSError:
            raise Exception('FSL environment variable not set.')

        if config.verbose:
            print('\n--- Environment Setup ---')

        if Analysis._brain_directory == "":

            if config.brain_file_loc in ("", " "):
                print('Select the directory of the raw MRI/fMRI brains.')

                Analysis._brain_directory = Utils.file_browser()

            else:
                Analysis._brain_directory = config.brain_file_loc

                if config.verbose:
                    print(f'Gathering brain files from {config.brain_file_loc}.')

            # Save copy of config.py to retain settings. It is saved here as after changing directory it will be harder
            # to find
            Utils.save_config(Analysis._brain_directory)

            try:
                os.chdir(Analysis._brain_directory)
            except FileNotFoundError:
                raise FileNotFoundError('brain_file_loc in config.py is not a valid directory.')

        Analysis._atlas_name = os.path.splitext(Analysis._atlas_label_list[int(config.atlas_number)][1])[0]
        if config.verbose:
            print('Using the ' + Analysis._atlas_name + ' atlas.')

        Analysis._save_location = Analysis._atlas_name + "_ROI_report/"

        # Find all nifti and analyze files
        Analysis.brain_file_list = Utils.find_files(Analysis._brain_directory, "hdr", "nii.gz", "nii")

        if len(Analysis.brain_file_list) == 0:
            raise NameError("No files found.")

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(Analysis._brain_directory + "/" + Analysis._save_location)

        Utils.move_file('config_log.py', Analysis._brain_directory,
                      Analysis._save_location)  # Move config file to analysis folder

        # Extract labels from selected FSL atlas
        Analysis.roi_label_list()

        brain_class_list = []
        for brain in Analysis.brain_file_list:
            # Initialise Analysis class for each file found
            brain_class_list.append(Analysis(brain, atlas=Analysis._atlas_name, atlas_path=Analysis._atlas_path,
                                             labels=Analysis._labelArray))

        return brain_class_list

    def run_analysis(self, brain_number_current, brain_number_total, freesurfer_excluded_voxels):
        if config.verbose:
            print(f'\nAnalysing brain {brain_number_current + 1}/{brain_number_total}: {self.brain}.\n')
        self.roi_flirt_transform()
        self.roi_stats(brain_number_current, brain_number_total)

        return self

    def roi_flirt_transform(self):
        """Function which uses NiPype to transform the chosen atlas into the native space."""
        if config.motion_correct:
            self.mcflirt()
        self.mean_over_time()
        self.brain_extraction()
        self.convert_to_mni()
        self.invert_transform()
        self.apply_inverse_transform()

    def mcflirt(self):
        mcflirt = fsl.MCFLIRT()
        mcflirt.inputs.in_file = self.brain
        self.mc_brain = mcflirt.inputs.out_file = self._save_location + self.no_ext_brain + '_mc.nii.gz'
        mcflirt.run()

        self.file_list.append(self.mc_brain)

    def mean_over_time(self):
        # If file is 4D, convert to 3D using "fslmaths -Tmean"
        tMean = fsl.MeanImage()

        if config.motion_correct:
            tMean.inputs.in_file = self.mc_brain
        else:
            tMean.inputs.in_file = self.brain

        self.mean_brain = tMean.inputs.out_file = self._save_location + self.no_ext_brain + '_mean.nii.gz'
        tMean.run()

        self.file_list.append(self.mean_brain)

    def brain_extraction(self):
        # Brain extraction
        bet = fsl.BET()
        bet.inputs.in_file = self.mean_brain
        bet.inputs.frac = config.frac_inten
        self.bet_brain = bet.inputs.out_file = self._save_location + self.no_ext_brain + '_bet.nii.gz'
        bet.run()

        self.file_list.append(self.bet_brain)

    def convert_to_mni(self):
        # Convert to MNI space
        flirt = fsl.FLIRT()  # TODO: Is there a better way than flirt?
        flirt.inputs.in_file = self.bet_brain
        flirt.inputs.reference = self._fsl_path + '/data/standard/MNI152_T1_1mm_brain.nii.gz'  # TODO: This doesnt work because it cant find the FSL path anymore
        flirt.inputs.dof = config.dof
        self.brain_to_mni = flirt.inputs.out_file = self._save_location + self.no_ext_brain + '_to_mni.nii.gz'
        self.brain_to_mni_mat = flirt.inputs.out_matrix_file = self._save_location + self.no_ext_brain + '_to_mni.mat'
        flirt.run()

        self.file_list.extend([self.brain_to_mni, self.brain_to_mni_mat])

    def invert_transform(self):
        invt = fsl.ConvertXFM()
        invt.inputs.in_file = self.brain_to_mni_mat
        invt.inputs.invert_xfm = True
        self.invt_mni_transform = invt.inputs.out_file = self._save_location + 'mni_to_' + self.no_ext_brain + '.mat'
        invt.run()

        self.file_list.append(self.invt_mni_transform)

    def apply_inverse_transform(self):
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

    def roi_stats(self, brain_number_current, brain_number_total, excluded_voxels=None):
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
                        if config.verbose:
                            print("This may take a while...")
                    if config.verbose:
                        print("  - Bootstrapping roi {}/{}".format(counter, roiNum))
                    roiResults[3, roi] = Utils.calculate_confidence_interval(roiTempStore,
                                                                             roi=roi)  # TODO: Speed up code, multithreading?
                # Calculate overall statistics
                roiResults[3, -1] = Utils.calculate_confidence_interval(
                    roiTempStore[start_val:, :])  # TODO does this work?

            headers = ['Voxels', 'Mean', 'Std_dev',
                       'Conf_Int_%s' % self._conf_level_list[int(config.conf_level_number)][0],
                       'Min', 'Max', 'Freesurfer excluded voxels']

            # Reorganise matrix to later remove nan rows
            roiTempStore = roiTempStore.transpose()
            i = np.arange(roiTempStore.shape[1])
            # Find indices of nans and put them at the end of each column
            a = np.isnan(roiTempStore).argsort(0, kind='mergesort')
            # Reorganise matrix with nans at end
            roiTempStore[:] = roiTempStore[a, i]

            # Save results as dataframe
            results = pd.DataFrame(data=roiResults,
                                   index=headers,
                                   columns=self.label_list)
            raw_results = pd.DataFrame(data=roiTempStore,
                                       columns=self.label_list[:-1])

            # Remove the required rows from the dataframe
            if exclude_rois:
                results = results.drop(results.columns[exclude_rois], axis=1)
                raw_results = raw_results.drop(raw_results.columns[exclude_rois], axis=1)

            if exclude_rois is None or 0 not in exclude_rois:
                raw_results = raw_results.drop(raw_results.columns[0], axis=1)

            # Remove rows where all columns have NaNs (essential to keep file size down)
            raw_results = raw_results.dropna(axis=0, how='all')

            raw_results_path = f"{self._brain_directory}/{self._save_location}Raw_results/"
            Utils.check_and_make_dir(raw_results_path)

            # Save JSON files
            if config.verbose:
                print(f'\nSaving JSON files for brain {brain_number_current + 1}/{brain_number_total}.\n')

            with open(self._save_location + self.no_ext_brain + ".json", 'w') as file:
                json.dump(results.to_dict(), file, indent=2)
            with open(raw_results_path + self.no_ext_brain + "_raw.json", 'w') as file:
                json.dump(raw_results.to_dict(), file, indent=2, ignore_nan=True)

            # Save variable for atlas_scale function
            self.roiResults = roiResults

        def nipype_file_cleanup():
            """Clean up unnecessary output."""
            if config.save_stats_only:
                for file in self.file_list:
                    os.remove(file)

        calculate_stats()
        nipype_file_cleanup()

    def atlas_scale(self, max_roi_stat, brain_number_current, brain_number_total):
        """Produces up to three scaled json files. Within brains, between brains (based on rois), between brains
        (based on the highest seen value of all brains and rois). Only the first json file will be created if using if
        running the analysis with all atlases."""
        if brain_number_current == 0:
            if config.verbose:
                print('\n--- Image creation ---')
        if config.verbose:
            print('\n Creating images for {brain}: {brain_num_cur}/{brain_num_tot}.\n'.format(
                brain_num_cur=brain_number_current + 1,
                brain_num_tot=brain_number_total,
                brain=self.brain))

        brain_stat = nib.load(self.atlas_path)
        brain_stat = brain_stat.get_fdata()

        within_roi_stat = deepcopy(brain_stat)
        mixed_roi_stat = deepcopy(brain_stat)

        roi_scaled_stat = [(y / x) * 100 for x, y in zip(max_roi_stat, self.roiResults[config.roi_stat_number, :])]
        global_scaled_stat = [(y / max(max_roi_stat)) * 100 for y in self.roiResults[config.roi_stat_number, :]]

        roi_stat_brain_size = brain_stat.shape

        # Iterate through each voxel in the atlas
        for x in range(0, roi_stat_brain_size[0]):
            for y in range(0, roi_stat_brain_size[1]):
                for z in range(0, roi_stat_brain_size[2]):
                    # Set new value of voxel to the required statistic
                    roi_row = int(brain_stat[x][y][z])
                    if roi_row == 0:
                        brain_stat[x][y][z] = np.nan
                        within_roi_stat[x][y][z] = np.nan
                        mixed_roi_stat[x][y][z] = np.nan
                    else:
                        brain_stat[x][y][z] = self.roiResults[config.roi_stat_number, roi_row]
                        within_roi_stat[x][y][z] = roi_scaled_stat[roi_row]
                        mixed_roi_stat[x][y][z] = global_scaled_stat[roi_row]

        # Convert atlas to NIFTI and save it
        affine = np.eye(4)
        scaled_atlas = nib.Nifti1Image(brain_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s.nii.gz"
                                 % self.atlas_scale_filename[config.roi_stat_number])

        scaled_atlas = nib.Nifti1Image(within_roi_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_within_roi_scaled.nii.gz"
                                 % self.atlas_scale_filename[config.roi_stat_number])

        scaled_atlas = nib.Nifti1Image(mixed_roi_stat, affine)
        scaled_atlas.to_filename(self._save_location + self.no_ext_brain + "_%s_mixed_roi_scaled.nii.gz"
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
    def print_info(cls):
        """Produce print_info text when the parameter "print_info" is passed with the filename. I.e. "analysis.py print_info"."""

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

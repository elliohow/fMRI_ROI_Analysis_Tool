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

from .utils import Utils

config = None


class Analysis:
    file_list = []

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
    _anat_brain = ""
    _anat_brain_to_mni = ""
    atlas_path = ""
    _atlas_label_path = ""
    _atlas_name = ""
    _labelArray = []

    def __init__(self, brain, atlas="", atlas_path="", labels=""):
        self.brain = brain
        self.label_list = labels
        self.atlas_path = atlas_path

        self.no_ext_brain = atlas + "_" + os.path.splitext(self.brain)[0]
        self.stat_brain = config.stat_map_folder + os.path.splitext(self.brain)[0] + config.stat_map_suffix

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
        self._atlas_label_path = self._atlas_label_path
        self._atlas_name = self._atlas_name
        self._labelArray = self._labelArray

        # These are imported later on using the save_class_variables function
        self._anat_brain = ""
        self._anat_brain_to_mni = ""

    @staticmethod
    def setup_analysis(cfg):
        """Set up environment and find files before running analysis."""

        global config
        config = cfg

        try:
            Analysis._fsl_path = os.environ['FSLDIR']
        except OSError:
            raise Exception('FSL environment variable not set.')

        if config.verbose:
            print('\n--- Environment Setup ---')

        if config.brain_file_loc in ("", " "):
            print('Select the directory of the raw MRI/fMRI brains.')

            Analysis._brain_directory = Utils.file_browser(title='Select the directory of the raw MRI/fMRI brains')

        else:
            Analysis._brain_directory = config.brain_file_loc

            if config.verbose:
                print(f'Gathering brain files from {config.brain_file_loc}.')

        # Save copy of config_log.toml to retain settings. It is saved here as after changing directory it
        # will be harder to find
        Utils.save_config(Analysis._brain_directory)

        try:
            os.chdir(Analysis._brain_directory)
        except FileNotFoundError:
            raise FileNotFoundError('brain_file_loc in config.toml is not a valid directory.')

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

        Utils.move_file('config_log.toml', Analysis._brain_directory,
                        Analysis._save_location)  # Move config file to analysis folder

        # Extract labels from selected FSL atlas
        Analysis.roi_label_list()

        brain_class_list = []
        for brain in Analysis.brain_file_list:
            # Initialise Analysis class for each file found
            brain_class_list.append(Analysis(brain, atlas=Analysis._atlas_name, atlas_path=Analysis.atlas_path,
                                             labels=Analysis._labelArray))

        return brain_class_list

    @classmethod
    def anat_setup(cls):
        if config.verbose:
            print('\nConverting anatomical file to MNI space.')

        anat = Utils.find_files(f'{os.getcwd()}/anat/', "hdr", "nii.gz", "nii")[0]

        cls._anat_brain = cls.fsl_functions(cls, cls._save_location, os.path.splitext(anat)[0],
                                            'BET', f'{os.getcwd()}/anat/{anat}', "bet_")

        cls._anat_brain_to_mni = cls.fsl_functions(cls, cls._save_location, f"bet_{os.path.splitext(anat)[0]}",
                                                   'FLIRT', cls._anat_brain, 'to_mni_from_',
                                                   f'{cls._fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')

    def save_class_variables(self):
        # Due to dill errors with saving class variables of imported classes, this function is used to save the class
        # variable as an instance variable, however it needs to be called after brain extraction of the anatomical
        # instantiation, which is why it isn't saved during instantiation.
        self._anat_brain = Analysis._anat_brain
        self._anat_brain_to_mni = Analysis._anat_brain_to_mni

    def run_analysis(self, brain_number_current, brain_number_total, cfg):
        global config
        config = cfg

        if config.verbose:
            print(f'\nAnalysing brain {brain_number_current + 1}/{brain_number_total}: {self.brain}.\n')

        excluded_voxels = self.roi_flirt_transform()  # Convert MNI brain to native space
        self.roi_stats(brain_number_current, brain_number_total, excluded_voxels)  # Calculate and save statistics

        return self

    def roi_flirt_transform(self):
        """Function which uses NiPype to transform the chosen atlas into native space."""
        pack_vars = [self, self._save_location, self.no_ext_brain]

        if config.motion_correct:
            # Motion correction
            self.fsl_functions(*pack_vars, 'MCFLIRT', self.brain, "mc_")

        # Turn 4D scan into 3D
        current_brain = self.fsl_functions(*pack_vars, 'MeanImage', self.brain, "mean_")

        # Brain extraction
        current_brain = self.fsl_functions(*pack_vars, 'BET', current_brain, "bet_", config.frac_inten)

        if config.anat_align:
            # Align to anatomical
            anat_aligned_mat = self.fsl_functions(*pack_vars, 'FLIRT', current_brain,  "to_anat_from_", self._anat_brain)

            # Combine fMRI-anat and anat-mni matrices
            mat = self.fsl_functions(*pack_vars, 'ConvertXFM', anat_aligned_mat, 'combined_mat_', 'concat_xfm')

        else:
            # Align to MNI
            mat = self.fsl_functions(*pack_vars, 'FLIRT', current_brain, "to_mni_from_",
                                     f'{self._fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')
        # Get inverse of matrix
        inverse_mat = self.fsl_functions(*pack_vars, 'ConvertXFM', mat, 'inverse_combined_mat')

        # Apply inverse of matrix to chosen atlas to convert it into standard space
        self.fsl_functions(*pack_vars, 'ApplyXFM', self.atlas_path, 'mni_to_', inverse_mat, current_brain, 'nearestneighbour')

        if config.grey_matter_segment is not None:
            # Convert segmentation to fMRI native space
            segmentation_to_fmri = self.segmentation_to_fmri(anat_aligned_mat, current_brain)

            return self.find_gm_from_segment(segmentation_to_fmri)

    def roi_stats(self, brain_number_current, brain_number_total, excluded_voxels):
        """Function which uses the output from the roi_flirt_transform function to collate the statistical information
        per ROI."""

        # Load brains and pre-initialise arrays
        roiTempStore, roiResults, idxMNI, idxBrain, roiList, roiNum = self.roi_stats_setup()

        # Combine information from fMRI and MNI brains (both in native space) to assign an ROI to each voxel
        roiTempStore, roiResults = self.calculate_voxel_stats(roiTempStore, roiResults, idxMNI, idxBrain, excluded_voxels)

        warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

        roiResults = self.calculate_roi_stats(roiTempStore, roiResults)  # Compile ROI statistics

        warnings.filterwarnings('default')  # Reactivate warnings

        if config.bootstrap:
            self.roi_stats_bootstrap(roiTempStore, roiResults, roiNum)  # Bootstrapping

        self.roi_stats_save(roiTempStore, roiResults, brain_number_current, brain_number_total)  # Save results

        self.roiResults = roiResults  # Retain variable for atlas_scale function

        self.file_cleanup(self.file_list, self._save_location) # Clean up files

    @staticmethod
    def fsl_functions(object, save_location, no_ext_brain, func, input, prefix, *argv):
        """Run an FSL function using NiPype."""
        fslfunc = getattr(fsl, func)()
        fslfunc.inputs.in_file = input
        fslfunc.inputs.output_type = 'NIFTI_GZ'

        if func == 'ConvertXFM':
            suffix = '.mat'
        else:
            suffix = '.nii.gz'

        current_brain = fslfunc.inputs.out_file = f"{save_location}{prefix}{no_ext_brain}{suffix}"

        # Arguments dependent on FSL function used
        if func == 'MCFLIRT':
            object.brain = current_brain  # TODO comment this

        elif func == 'BET':
            fslfunc.inputs.frac = config.frac_inten
            fslfunc.inputs.functional = True

        elif func == 'FLIRT':
            fslfunc.inputs.reference = argv[0]
            fslfunc.inputs.dof = config.dof
            current_mat = fslfunc.inputs.out_matrix_file = f'{save_location}{prefix}{no_ext_brain}.mat'

        elif func == 'ConvertXFM':
            if len(argv) > 0 and argv[0] == 'concat_xfm':
                fslfunc.inputs.in_file2 = object._anat_brain_to_mni
                fslfunc.inputs.concat_xfm = True
            else:
                fslfunc.inputs.invert_xfm = True

        elif func == 'ApplyXFM':
            fslfunc.inputs.apply_xfm = True
            fslfunc.inputs.in_matrix_file = argv[0]
            fslfunc.inputs.reference = argv[1]
            fslfunc.inputs.interp = argv[2]
            current_mat = fslfunc.inputs.out_matrix_file = f"{save_location}{prefix}{no_ext_brain}.mat"

        if config.verbose_cmd_line_args:
            print(fslfunc.cmdline)

        fslfunc.run()

        if func in ('FLIRT', 'ApplyXFM'):
            object.file_list.extend([current_brain, current_mat])

        elif func == 'BET':
            object.file_list.extend([current_brain, f"{save_location}{prefix}{no_ext_brain}_mask{suffix}"])

        else:
            object.file_list.append(current_brain)

        if func == 'FLIRT':
            return current_mat

        return current_brain

    def segmentation_to_fmri(self, anat_aligned_mat, current_brain):
        if config.verbose:
            print(f'- Aligning {config.grey_matter_segment} segmentation to fMRI volume: {self.brain}')

        if config.grey_matter_segment == "freesurfer":
            source_loc = 'freesurfer/mri/native_segmented_brain.nii'  # Use anat aligned freesurfer segmentation
            prefix = 'freesurf_to_'
            interp = 'nearestneighbor'

        elif config.grey_matter_segment == "fslfast":
            anat_brain_no_folder = os.path.split(self._anat_brain)[-1]
            anat_brain_base = os.path.splitext(os.path.splitext(anat_brain_no_folder)[0])[0]
            source_loc = f'fslfast/{anat_brain_base}_pve_1.nii.gz'
            prefix = 'fslfast_to_'
            interp = 'trilinear'

        # Save inverse of fMRI to anat
        inverse_mat = self.fsl_functions(self, self._save_location, self.no_ext_brain, 'ConvertXFM', anat_aligned_mat,
                                         'inverse_anat_to_')

        # Apply inverse of matrix to chosen segmentation to convert it into native space
        segmentation_to_fmri = self.fsl_functions(self, self._save_location, self.no_ext_brain, 'ApplyXFM', source_loc,
                                                  prefix, inverse_mat, current_brain, interp)

        return segmentation_to_fmri

    @staticmethod
    def find_gm_from_segment(native_space_segment):
        segment_brain = nib.load(native_space_segment)
        segment_brain = segment_brain.get_fdata().flatten()

        # Make a list that will return 0 for each voxel that is not csf or wm
        idxCSF_or_WM = np.full([segment_brain.shape[0]], 0)

        if config.grey_matter_segment == 'freesurfer':
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

            # If voxel has a value found in the list above then set to 1
            for counter, voxel in enumerate(segment_brain):
                if voxel in csf_wm_values:
                    idxCSF_or_WM[counter] = 1

        elif config.grey_matter_segment == 'fslfast':
            # If voxel has a value below the threshold then set to 1
            idxCSF_or_WM = (segment_brain < config.fslfast_min_prob).astype(int)

        return idxCSF_or_WM

    def roi_stats_setup(self):
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
        roiResults[6, 0:-1] = 0  # Change excluded voxels measure from NaN to 0

        return roiTempStore, roiResults, idxMNI, idxBrain, roiList, roiNum

    @staticmethod
    def calculate_voxel_stats(roiTempStore, roiResults, idxMNI, idxBrain, excluded_voxels):
        for counter, roi in enumerate(idxMNI):
            if config.grey_matter_segment is None or excluded_voxels[counter] == 0:
                roiTempStore[int(roi), counter] = idxBrain[counter]

            else:
                roiTempStore[0, counter] = idxBrain[counter]  # Assign to No ROI if voxel is excluded
                roiResults[6, int(roi)] += 1

        return roiTempStore, roiResults

    def calculate_roi_stats(self, roiTempStore, roiResults):
        axis = 1
        write_start = 0
        write_end = -1
        read_start = 0

        for i in range(2):
            roiResults[0, write_start:write_end] = np.count_nonzero(~np.isnan(roiTempStore[read_start:, :]), axis=axis)  # Count number of non-nan voxels
            roiResults[1, write_start:write_end] = np.nanmean(roiTempStore[read_start:, :], axis=axis)
            roiResults[2, write_start:write_end] = np.nanstd(roiTempStore[read_start:, :], axis=axis)
            roiResults[3, write_start:write_end] = self._conf_level_list[int(config.conf_level_number)][1] \
                                                   * roiResults[2, write_start:write_end] \
                                                   / np.sqrt(roiResults[0, write_start:write_end])  # 95% confidence interval calculation
            roiResults[4, write_start:write_end] = np.nanmin(roiTempStore[read_start:, :], axis=axis)
            roiResults[5, write_start:write_end] = np.nanmax(roiTempStore[read_start:, :], axis=axis)

            axis = None
            read_start = 1
            write_start = -1
            write_end = None

        roiResults[6, -1] = np.sum(roiResults[6, 1:-1])  # Calculate excluded voxels

        # Convert ROIs with no voxels from columns with NaNs to zeros
        for column, voxel_num in enumerate(roiResults[0]):
            if voxel_num == 0.0:
                for row in list(range(1, 6)):
                    roiResults[row][column] = 0.0

        return roiResults

    def roi_stats_bootstrap(self, roiTempStore, roiResults, roiNum):
        for counter, roi in enumerate(list(range(0, roiNum + 1))):
            if config.verbose:
                print(f"  - Bootstrapping ROI {counter + 1}/{roiNum + 1}: {self.brain}.")

            if counter < roiNum:
                roiResults[1, roi], roiResults[3, roi] = Utils.calculate_confidence_interval(roiTempStore,
                                                                                             config.bootstrap_alpha,
                                                                                             roi=roi)
            else:
                # Calculate overall statistics
                roiResults[1, -1], roiResults[3, -1] = Utils.calculate_confidence_interval(roiTempStore[1:, :],
                                                                                           config.bootstrap_alpha)

    def roi_stats_save(self, roiTempStore, roiResults, brain_number_current, brain_number_total):
        headers = ['Voxels', 'Mean', 'Std_dev',
                   f'Conf_Int_{self._conf_level_list[int(config.conf_level_number)][0]}',
                   'Min', 'Max', 'Excluded_Voxels']

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
        raw_results = raw_results.drop(raw_results.columns[0], axis=1)

        # Remove rows where all columns have NaNs (essential to keep file size down)
        raw_results = raw_results.dropna(axis=0, how='all')

        summary_results_path = f"{self._brain_directory}/{self._save_location}Summarised_results/"
        Utils.check_and_make_dir(summary_results_path)

        raw_results_path = f"{self._brain_directory}/{self._save_location}Raw_results/"
        Utils.check_and_make_dir(raw_results_path)

        # Save JSON files
        if config.verbose:
            print(f'\nSaving JSON files for brain {brain_number_current + 1}/{brain_number_total}: {self.brain}.\n')

        with open(summary_results_path + self.no_ext_brain + ".json", 'w') as file:
            json.dump(results.to_dict(), file, indent=2)
        with open(raw_results_path + self.no_ext_brain + "_raw.json", 'w') as file:
            json.dump(raw_results.to_dict(), file, indent=2, ignore_nan=True)

    @staticmethod
    def file_cleanup(file_list, save_location):
        """Clean up unnecessary output."""
        if config.file_cleanup == 'delete':
            for file in file_list:
                os.remove(file)

        elif config.file_cleanup == 'move':
            Utils.check_and_make_dir(f"{save_location}Intermediate_files")

            for file in file_list:
                file = file.replace(save_location, "")  # Remove folder from start of file name
                Utils.move_file(file, save_location, f"{save_location}Intermediate_files")

    def atlas_scale(self, max_roi_stat, brain_number_current, brain_number_total, config):
        """Produces up to three scaled NIFTI files. Within brains, between brains (based on rois), between brains
        (based on the highest seen value of all brains and rois)."""
        if brain_number_current == 0:
            if config.verbose:
                print('\n--- Atlas scaling ---')
        if config.verbose:
            print('\n Creating NIFTI_ROI files for {brain}: {brain_num_cur}/{brain_num_tot}.\n'.format(
                brain_num_cur=brain_number_current + 1,
                brain_num_tot=brain_number_total,
                brain=self.brain))

        brain_stat = nib.load(self.atlas_path)
        brain_stat = brain_stat.get_fdata()

        within_roi_stat = deepcopy(brain_stat)
        mixed_roi_stat = deepcopy(brain_stat)

        np.seterr('ignore')  # Ignore runtime warning when dividing by 0 (where ROIs have been excluded)
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
        scale_stat = [
                (brain_stat,
                 f"{self.no_ext_brain}_{self.atlas_scale_filename[config.roi_stat_number]}.nii.gz"),
                (within_roi_stat,
                 f"{self.no_ext_brain}_{self.atlas_scale_filename[config.roi_stat_number]}_within_roi_scaled.nii.gz"),
                (mixed_roi_stat,
                 f"{self.no_ext_brain}_{self.atlas_scale_filename[config.roi_stat_number]}_mixed_roi_scaled.nii.gz")
                    ]

        scaled_brains = []

        for i in scale_stat:
            scaled_brain = nib.Nifti1Image(i[0], affine)
            scaled_brain.to_filename(f"{self._save_location}{i[1]}")

            scaled_brains.append(i[1])

        for brain in scaled_brains:
            Utils.move_file(brain, f"{os.getcwd()}/{self._save_location}",
                            f"{os.getcwd()}/{self._save_location}NIFTI_ROI")

    @classmethod
    def roi_label_list(cls):
        """Extract labels from specified FSL atlas XML file."""
        cls.atlas_path = cls._fsl_path + '/data/atlases/' + cls._atlas_label_list[int(config.atlas_number)][0]
        cls._atlas_label_path = cls._fsl_path + '/data/atlases/' + cls._atlas_label_list[int(config.atlas_number)][1]

        with open(cls._atlas_label_path) as fd:
            atlas_label_dict = xmltodict.parse(fd.read())

        cls._labelArray = []
        cls._labelArray.append('No ROI')

        for roiLabelLine in atlas_label_dict['atlas']['data']['label']:
            cls._labelArray.append(roiLabelLine['#text'])

        cls._labelArray.append('Overall')

    @classmethod
    def freesurfer_to_anat(cls):
        """Function which removes freesurfer padding and transforms freesurfer segmentation to native space."""
        if config.verbose:
            print('Aligning freesurfer file to anatomical native space.')
        # Rawavg is in native anatomical space, so align to this file. vol_label_file defines output file name.
        native_segmented_brain = freesurfer.Label2Vol(seg_file='freesurfer/mri/aseg.auto_noCCseg.mgz',
                                                      template_file='freesurfer/mri/rawavg.mgz',
                                                      vol_label_file='freesurfer/mri/native_segmented_brain.mgz',
                                                      reg_header='freesurfer/mri/aseg.auto_noCCseg.mgz',
                                                      terminal_output='none')
        native_segmented_brain.run()

        mgz_to_nii = freesurfer.MRIConvert(in_file='freesurfer/mri/native_segmented_brain.mgz',
                                           out_file='freesurfer/mri/native_segmented_brain.nii',
                                           out_type='nii',
                                           terminal_output='none')
        mgz_to_nii.run()

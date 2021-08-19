import itertools
import os
import re
import sys
import warnings
from copy import deepcopy

import nibabel as nib
import numpy as np
import pandas as pd
import simplejson as json
import xmltodict
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from os.path import splitext
from glob import glob
from nipype.interfaces import fsl
from sklearn.metrics import mean_squared_error as mse

from .utils import Utils
from .paramparser import ParamParser

config = None


class Environment_Setup:
    file_list = []
    save_location = None
    roi_stat_list = ["Voxel number", "Mean", "Standard Deviation", "Confidence Interval", "Min", "Max"]
    conf_level_list = [('80', 1.28),
                       ('85', 1.44),
                       ('90', 1.64),
                       ('95', 1.96),
                       ('98', 2.33),
                       ('99', 2.58)]
    atlas_label_list = [('Cerebellum/Cerebellum-MNIflirt-maxprob-thr0-1mm.nii.gz', 'Cerebellum_MNIflirt.xml'),
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
    base_directory = None
    fsl_path = None
    atlas_path = None
    atlas_label_path = None
    atlas_name = None
    labelArray = []

    @classmethod
    def setup_analysis(cls, cfg):
        """Set up environment and find files before running analysis."""
        global config
        config = cfg

        if config.verbose:
            print('\n--- Environment Setup ---')

        cls.setup_environment(config)

        if config.verify_param_method and config.run_plotting == 'table':
            verify_param_values()  # TODO: check this works with participant folders

        cls.setup_save_location(config)

        # Extract labels from selected FSL atlas
        cls.roi_label_list()

        participant_list = cls.setup_participants()

        matched_brain_list = cls.matched_brains_setup(participant_list)

        return participant_list, matched_brain_list

    @classmethod
    def matched_brains_setup(cls, participant_list):
        matched_brains = cls.find_shared_params(participant_list)  # Find brains which share parameter combinations TODO: Raise error if paramvalues is different

        MatchedBrains.critical_parameters = config.parameter_dict1

        matched_brain_list = set()
        for combination in matched_brains:
            # Initialise participants
            matched_brain_list.add(MatchedBrains(matched_brains[combination], combination))

        return matched_brain_list

    @classmethod
    def find_shared_params(cls, participant_list):
        param_values = ParamParser.load_paramValues_file()

        matched_brains = dict()
        for index, row in param_values.iterrows():
            if tuple(row[2:-1]) not in matched_brains.keys():
                matched_brains[tuple(row[2:-1])] = [cls.find_brain_object(row, participant_list)]
            else:
                matched_brains[tuple(row[2:-1])].append(cls.find_brain_object(row, participant_list))

        return matched_brains

    @staticmethod
    def find_brain_object(row, participant_list):
        participant = next(participant for participant in participant_list
                           if participant.participant_name == row['Participant'])
        brain = next(brain for brain in participant.brains
                     if brain.no_ext_brain == row['File name'])

        return {'Participant': participant.participant_name, 'Brain': brain.no_ext_brain}


    @classmethod
    def setup_participants(cls):
        participant_dirs = cls.find_participant_dirs()

        participant_list = set()
        for participant_dir in participant_dirs:
            # Initialise participants
            participant_list.add(Participant(participant_dir, cls.base_directory, cls.save_location))

        for participant in participant_list:
            participant.setup()

        return participant_list

    @classmethod
    def find_participant_dirs(cls):
        participant_dirs = [direc for direc in glob("*") if re.search("^p[0-9]+", direc)]

        if len(participant_dirs) == 0:
            raise FileNotFoundError('Participant directories not found.')
        elif config.verbose:
            print(f'Found {len(participant_dirs)} participant folders.')
            
        return participant_dirs

    @classmethod
    def setup_save_location(cls, config):
        cls.atlas_name = os.path.splitext(cls.atlas_label_list[int(config.atlas_number)][1])[0]
        
        if config.output_folder == 'DEFAULT':
            cls.save_location = f'{cls.atlas_name}_ROI_report/'
        else:
            cls.save_location = f'{config.output_folder}/'
            
        if config.verbose:
            print(f'Using the {cls.atlas_name} atlas.'
                  f'\n Saving output in directory: {cls.save_location}')

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(f"{cls.base_directory}/{cls.save_location}")

        # Move config file to analysis folder
        Utils.move_file('config_log.toml', cls.base_directory, cls.save_location)

    @classmethod
    def setup_environment(cls, config):
        try:
            cls.fsl_path = os.environ['FSLDIR']
        except OSError:
            raise Exception('FSL environment variable not set.')

        if config.brain_file_loc in ("", " "):
            print('Select the directory of the raw MRI/fMRI brains.')
            cls.base_directory = Utils.file_browser(title='Select the directory of the raw MRI/fMRI brains')

        else:
            cls.base_directory = config.brain_file_loc

            if config.verbose:
                print(f'Gathering brain files from {config.brain_file_loc}.')

        # Save copy of config_log.toml to retain settings. It is saved here as after changing directory it will be harder to find
        Utils.save_config(cls.base_directory)

        try:
            os.chdir(cls.base_directory)
        except FileNotFoundError:
            raise FileNotFoundError('brain_file_loc in config.toml is not a valid directory.')

    @classmethod
    def roi_label_list(cls):
        """Extract labels from specified FSL atlas XML file."""
        cls.atlas_path = f'{cls.fsl_path}/data/atlases/{cls.atlas_label_list[int(config.atlas_number)][0]}'
        cls.atlas_label_path = f'{cls.fsl_path}/data/atlases/{cls.atlas_label_list[int(config.atlas_number)][1]}'

        with open(cls.atlas_label_path) as fd:
            atlas_label_dict = xmltodict.parse(fd.read())

        cls.labelArray = []
        cls.labelArray.append('No ROI')

        for roiLabelLine in atlas_label_dict['atlas']['data']['label']:
            cls.labelArray.append(roiLabelLine['#text'])

        cls.labelArray.append('Overall')


class Participant:
    def __init__(self, participant, base_directory, save_location):
        self.participant_name = participant
        self.participant_path = f"{base_directory}/{participant}"
        self.save_location = f"{self.participant_path}/{save_location}"
        self.anat_cost = 0
        self.brains = None
        self.file_list = []
        self.anat_brain = None
        self.anat_brain_to_mni = None
        self.anat_brain_no_ext = None

    def setup(self):
        self.find_fmri_files()

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(self.save_location)

        if config.anat_align:
            self.anat_setup()

        # Initialise brain class for each file found
        for counter, brain in enumerate(self.brains):
            self.brains[counter] = Brain(f"{self.participant_path}/fmri/{brain}",
                                         self.participant_path,
                                         self.participant_name,
                                         self.save_location,
                                         self.anat_brain,
                                         self.anat_brain_to_mni)

    def run_analysis(self, pool):
        if config.verbose:
            print(f'\nAnalysing fMRI files for participant: {self.participant_name}\n')

        # Set arguments to pass to run_analysis function
        iterable = zip(self.brains, itertools.repeat("run_analysis"), range(len(self.brains)),
                       itertools.repeat(len(self.brains)), itertools.repeat(config))

        if config.multicore_processing:
            brain_list = pool.starmap(Utils.instance_method_handler, iterable)
        else:
            brain_list = list(itertools.starmap(Utils.instance_method_handler, iterable))

        if config.anat_align:
            self.anat_file_cleanup()

        return brain_list

    def find_fmri_files(self):
        # Find all nifti and analyze files
        self.brains = Utils.find_files(f"{self.participant_path}/fmri", "hdr", "nii.gz", "nii")

        if len(self.brains) == 0:
            raise NameError("No files found.")

    def anat_setup(self):
        if config.verbose:
            print(f'\nConverting anatomical file to MNI space for participant: {self.participant_name}')

        anat = Utils.find_files(f'{self.participant_path}/anat/', "hdr", "nii.gz", "nii")

        if len(anat) > 1:
            raise FileExistsError('Multiple files found in anat folder.')
        else:
            anat = anat[0]

        self.anat_brain = f'{self.participant_path}/anat/{anat}'
        self.anat_brain_no_ext = anat.rsplit(".")[0]
        self.anat_brain_to_mni = fsl_functions(self, self.save_location, self.anat_brain_no_ext,
                                                'FLIRT', self.anat_brain, 'to_mni_from_',
                                                f'{Environment_Setup.fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')

    def calculate_anat_flirt_cost_function(self):
        fslfunc = fsl.FLIRT(in_file=self.anat_brain,
                            schedule=f'{Environment_Setup.fsl_path}/etc/flirtsch/measurecost1.sch',
                            terminal_output='allatonce', dof=config.dof)

        # Calculate MNI cost function value
        mni_cost = run_flirt_cost_function(fslfunc,
                                                f'{Environment_Setup.fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz',
                                                f'{self.save_location}Intermediate_files/to_mni_from_{self.anat_brain_no_ext}.mat',
                                                f'{self.save_location}Intermediate_files/{self.anat_brain_no_ext}_mni_redundant.nii.gz',
                                                f'{self.save_location}Intermediate_files/{self.anat_brain_no_ext}_mni_redundant.mat')

        return mni_cost

    def anat_file_cleanup(self):
        """Clean up unnecessary output from either instance of class, or class itself."""
        if config.file_cleanup == 'delete':
            for file in self.file_list:
                os.remove(file)

        elif config.file_cleanup == 'move':
            for file in self.file_list:
                file = file.replace(self.save_location, "")  # Remove folder from start of file name
                Utils.move_file(file, self.save_location, f"{self.save_location}Intermediate_files")

        self.file_list = []


class Brain:
    def __init__(self, brain, participant_folder, participant_name, save_location, anat_brain, anat_brain_to_mni):
        self.brain = brain
        self.save_location = save_location
        self.anat_brain = anat_brain
        self.anat_brain_to_mni = anat_brain_to_mni
        self.no_ext_brain = splitext(self.brain.split('/')[-1])[0]
        self.stat_brain = f"{participant_folder}/{config.stat_map_folder}{self.no_ext_brain}{config.stat_map_suffix}"
        self.roiResults = None
        self.roiTempStore = None
        self.roi_stat_list = ""
        self.file_list = []
        self.participant_name = participant_name

        # Copying class attributes here is a workaround for dill,
        # which can't access modified class attributes for imported modules.
        self._brain_directory = Environment_Setup.base_directory
        self._fsl_path = Environment_Setup.fsl_path
        self._atlas_label_path = Environment_Setup.atlas_label_path
        self._atlas_name = Environment_Setup.atlas_name
        self._labelArray = Environment_Setup.labelArray
        self.atlas_path = Environment_Setup.atlas_path

    def calculate_fmri_flirt_cost_function(self, brain_number_current, brain_number_total, config):
        if config.verbose:
            print(f'Calculating cost function value for {brain_number_current + 1}/{brain_number_total}: '
                  f'{self.no_ext_brain}')

        fslfunc = fsl.FLIRT(in_file=f'{self.save_location}Intermediate_files/bet_{self.no_ext_brain}.nii.gz',
                            schedule=f'{self._fsl_path}/etc/flirtsch/measurecost1.sch',
                            terminal_output='allatonce', dof=config.dof)

        anat_cost, mni_cost = 0, 0

        if config.anat_align:  # Calculate anatomical cost function value
            anat_cost = run_flirt_cost_function(fslfunc,
                                                          self.anat_brain,
                                                          f'{self.save_location}Intermediate_files/to_anat_from_{self.no_ext_brain}.mat',
                                                          f'{self.save_location}Intermediate_files/{self.no_ext_brain}_anat_redundant.nii.gz',
                                                          f'{self.save_location}Intermediate_files/{self.no_ext_brain}_anat_redundant.mat',
                                                     config)

        else:
            # Calculate MNI cost function value
            mni_cost = run_flirt_cost_function(fslfunc,
                                                         f'{self._fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz',
                                                         f'{self.save_location}/Intermediate_files/to_mni_from_{self.no_ext_brain}.mat',
                                                         f'{self.save_location}/Intermediate_files/{self.no_ext_brain}_mni_redundant.nii.gz',
                                                         f'{self.save_location}/Intermediate_files/{self.no_ext_brain}_mni_redundant.mat',
                                                    config)

        return self.no_ext_brain, anat_cost, mni_cost

    def run_analysis(self, brain_number_current, brain_number_total, cfg):
        global config
        config = cfg

        if config.verbose:
            print(f'Analysing fMRI volume {brain_number_current + 1}/{brain_number_total}: {self.no_ext_brain}')

        GM_bool = self.roi_flirt_transform(brain_number_current,
                                           brain_number_total)  # Convert MNI brain to native space

        self.roi_stats(brain_number_current, brain_number_total, GM_bool)  # Calculate and save statistics

        self.file_cleanup()  # Clean up files

        return self

    def file_cleanup(self):
        """Clean up unnecessary output from either instance of class, or class itself."""
        if config.file_cleanup == 'delete':
            for file in self.file_list:
                os.remove(file)

        elif config.file_cleanup == 'move':
            Utils.check_and_make_dir(f"{self.save_location}Intermediate_files")

            for file in self.file_list:
                file = file.replace(self.save_location, "")  # Remove folder from start of file name
                Utils.move_file(file, self.save_location, f"{self.save_location}Intermediate_files")

        self.file_list = []

    @staticmethod
    def roi_stats_bootstrap(roiTempStore, roiResults, roiNum, brain_number_current, brain_number_total):
        for counter, roi in enumerate(list(range(0, roiNum + 1))):
            if config.verbose:
                print(f"  - Bootstrapping ROI {counter + 1}/{roiNum + 1} "
                      f"for fMRI volume {brain_number_current + 1}/{brain_number_total}.")

            if counter < roiNum:
                roiResults[1, roi], roiResults[3, roi] = calculate_confidence_interval(roiTempStore,
                                                                                       config.bootstrap_alpha,
                                                                                       roi=roi)
            else:
                # Calculate overall statistics
                roiResults[1, -1], roiResults[3, -1] = calculate_confidence_interval(roiTempStore[1:, :],
                                                                                     config.bootstrap_alpha)
        return roiResults

    def roi_stats_save(self, roiTempStore, roiResults, brain_number_current, brain_number_total):
        headers = ['Voxels', 'Mean', 'Std_dev',
                   f'Conf_Int_{Environment_Setup.conf_level_list[int(config.conf_level_number)][0]}',
                   'Median', 'Min', 'Max', 'Excluded_Voxels']

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
                               columns=self._labelArray)
        raw_results = pd.DataFrame(data=roiTempStore,
                                   columns=self._labelArray[:-1])

        # Remove the required rows from the dataframe
        raw_results = raw_results.drop(raw_results.columns[0], axis=1)

        # Remove rows where all columns have NaNs (essential to keep file size down)
        raw_results = raw_results.dropna(axis=0, how='all')

        # Convert to dict and get rid of row numbers to significantly decrease file size
        roidict = Utils.dataframe_to_dict(raw_results)

        summary_results_path = f"{self.save_location}Summarised_results/"
        Utils.check_and_make_dir(summary_results_path)

        raw_results_path = f"{self.save_location}Raw_results/"
        Utils.check_and_make_dir(raw_results_path)

        # Save JSON files
        if config.verbose:
            print(f'Saving JSON files for fMRI volume {brain_number_current + 1}/{brain_number_total}.')

        with open(f"{summary_results_path}{self.no_ext_brain}.json", 'w') as file:
            json.dump(results.to_dict(), file, indent=2)
        with open(f"{raw_results_path}{self.no_ext_brain}_raw.json", 'w') as file:
            json.dump(roidict, file, indent=2)

    def roi_stats_setup(self):
        # Load original brain (with statistical map)
        stat_brain = nib.load(self.stat_brain)
        # Load atlas brain (which has been converted into native space)
        mni_brain = nib.load(f"{self.save_location}mni_to_{self.no_ext_brain}.nii.gz")

        stat_brain = stat_brain.get_fdata()
        mni_brain = mni_brain.get_fdata()

        if mni_brain.shape != stat_brain.shape:
            raise Exception('The matrix dimensions of the standard space and the statistical map brain do not '
                            'match.')

        # Find the number of unique ROIs in the atlas
        roiList = list(range(0, len(self._labelArray) - 1))
        roiNum = np.size(roiList)

        idxBrain = stat_brain.flatten()
        idxMNI = mni_brain.flatten()

        # Create arrays to store the values before and after statistics
        roiTempStore = np.full([roiNum, idxMNI.shape[0]], np.nan)
        roiResults = np.full([8, roiNum + 1], np.nan)
        roiResults[7, 0:-1] = 0  # Change excluded voxels measure from NaN to 0

        return roiTempStore, roiResults, idxMNI, idxBrain, roiList, roiNum

    @staticmethod
    def calculate_voxel_stats(roiTempStore, roiResults, idxMNI, idxBrain, GM_bool):
        for counter, roi in enumerate(idxMNI):
            if not config.grey_matter_segment or GM_bool[counter] == 1:
                roiTempStore[int(roi), counter] = idxBrain[counter]

            else:
                roiTempStore[0, counter] = idxBrain[counter]  # Assign to No ROI if voxel is excluded

                if int(roi) != 0:  # If ROI is not 'No ROI'
                    roiResults[7, int(roi)] += 1

        return roiTempStore, roiResults

    def segmentation_to_fmri(self, anat_aligned_mat, current_brain, brain_number_current, brain_number_total):
        if config.verbose:
            print(f'Aligning fslfast segmentation to fMRI volume {brain_number_current + 1}/{brain_number_total}.')

        try:
            source_loc = glob(f"fslfast/*_pve_1*")[0]
        except IndexError:
            source_loc = glob(f"fslfast/*")[0]

        prefix = 'fslfast_to_'
        interp = 'trilinear'

        # Save inverse of fMRI to anat
        inverse_mat = fsl_functions(self, self.save_location, self.no_ext_brain, 'ConvertXFM', anat_aligned_mat,
                                         'inverse_anat_to_')

        # Apply inverse of matrix to chosen segmentation to convert it into native space
        segmentation_to_fmri = fsl_functions(self, self.save_location, self.no_ext_brain, 'ApplyXFM', source_loc,
                                                  prefix, inverse_mat, current_brain, interp)

        return segmentation_to_fmri

    @staticmethod
    def find_gm_from_segment(native_space_segment):
        segment_brain = nib.load(native_space_segment)
        segment_brain = segment_brain.get_fdata().flatten()

        # If voxel has a value below the threshold then set to 1
        grey_matter_bool = (segment_brain >= config.fslfast_min_prob).astype(int)

        return grey_matter_bool

    def roi_flirt_transform(self, brain_number_current, brain_number_total):
        """Function which uses NiPype to transform the chosen atlas into native space."""
        pack_vars = [self, self.save_location, self.no_ext_brain]

        if config.motion_correct:
            # Motion correction
            fsl_functions(*pack_vars, 'MCFLIRT', self.brain, "mc_")

        # Turn 4D scan into 3D
        current_brain = fsl_functions(*pack_vars, 'MeanImage', self.brain, "mean_")

        # Brain extraction
        current_brain = fsl_functions(*pack_vars, 'BET', current_brain, "bet_")

        if config.anat_align:
            if config.verbose:
                print(f'Aligning fMRI volume {brain_number_current + 1}/{brain_number_total} to anatomical volume.')

            # Align to anatomical
            anat_aligned_mat = fsl_functions(*pack_vars, 'FLIRT', current_brain, "to_anat_from_", self.anat_brain)

            # Combine fMRI-anat and anat-mni matrices
            mat = fsl_functions(*pack_vars, 'ConvertXFM', anat_aligned_mat, 'combined_mat_', 'concat_xfm')

        else:
            # Align to MNI
            mat = fsl_functions(*pack_vars, 'FLIRT', current_brain, "to_mni_from_",
                                     f'{self._fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')

        # Get inverse of matrix
        inverse_mat = fsl_functions(*pack_vars, 'ConvertXFM', mat, 'inverse_combined_mat_')

        # Apply inverse of matrix to chosen atlas to convert it into native space
        fsl_functions(*pack_vars, 'ApplyXFM', self.atlas_path,
                      'mni_to_', inverse_mat, current_brain,
                      'nearestneighbour')

        if config.grey_matter_segment:
            # Convert segmentation to fMRI native space
            segmentation_to_fmri = self.segmentation_to_fmri(anat_aligned_mat, current_brain,
                                                             brain_number_current, brain_number_total)

            grey_matter_bool = self.find_gm_from_segment(segmentation_to_fmri)

            return grey_matter_bool

    def roi_stats(self, brain_number_current, brain_number_total, GM_bool):
        """Function which uses the output from the roi_flirt_transform function to collate the statistical information
        per ROI."""

        # Load brains and pre-initialise arrays
        roiTempStore, roiResults, idxMNI, idxBrain, roiList, roiNum = self.roi_stats_setup()

        # Combine information from fMRI and MNI brains (both in native space) to assign an ROI to each voxel
        roiTempStore, roiResults = self.calculate_voxel_stats(roiTempStore, roiResults,
                                                              idxMNI, idxBrain,
                                                              GM_bool)

        warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

        roiResults = compile_roi_stats(roiTempStore, roiResults)  # Compile ROI statistics

        warnings.filterwarnings('default')  # Reactivate warnings

        if config.bootstrap:
            roiResults = self.roi_stats_bootstrap(roiTempStore, roiResults, roiNum, brain_number_current,
                                                  brain_number_total)  # Bootstrapping

        self.roi_stats_save(roiTempStore, roiResults, brain_number_current, brain_number_total)  # Save results

        self.roiResults = roiResults  # Retain variable for atlas_scale function
        self.roiTempStore = roiTempStore  # Retain variable for atlas_scale function

    def atlas_scale(self, max_roi_stat, brain_number_current, brain_number_total, statistic_num, config):
        """Produces up to three scaled NIFTI files. Within brains, between brains (based on rois), between brains
        (based on the highest seen value of all brains and rois)."""
        if config.verbose:
            print(f'Creating {config.statistic_options[statistic_num]} NIFTI_ROI files for fMRI volume '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.brain}.')

        brain_stat = nib.load(Environment_Setup.atlas_path)
        brain_stat = brain_stat.get_fdata()

        within_roi_stat = deepcopy(brain_stat)
        mixed_roi_stat = deepcopy(brain_stat)

        np.seterr('ignore')  # Ignore runtime warning when dividing by 0 (where ROIs have been excluded)
        roi_scaled_stat = [(y / x) * 100 for x, y in zip(max_roi_stat, self.roiResults[statistic_num, :])]
        # Find maximum statistic value (excluding No ROI and overall category)
        global_scaled_stat = [(y / max(max_roi_stat[1:-1])) * 100 for y in self.roiResults[statistic_num, :]]

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
                        brain_stat[x][y][z] = self.roiResults[statistic_num, roi_row]
                        within_roi_stat[x][y][z] = roi_scaled_stat[roi_row]
                        mixed_roi_stat[x][y][z] = global_scaled_stat[roi_row]

        # Convert atlas to NIFTI and save it
        affine = np.eye(4)
        scale_stats = [
            (brain_stat,
             f"{self.no_ext_brain}_{config.statistic_options[statistic_num]}.nii.gz"),
            (within_roi_stat,
             f"{self.no_ext_brain}_{config.statistic_options[statistic_num]}_within_roi_scaled.nii.gz"),
            (mixed_roi_stat,
             f"{self.no_ext_brain}_{config.statistic_options[statistic_num]}_mixed_roi_scaled.nii.gz")
        ]

        scaled_brains = []

        for scale_stat in scale_stats:
            scaled_brain = nib.Nifti1Image(scale_stat[0], affine)
            scaled_brain.to_filename(f"{self.save_location}{scale_stat[1]}")

            scaled_brains.append(scale_stat[1])

        for brain in scaled_brains:
            Utils.move_file(brain, f"{os.getcwd()}/{self.save_location}",
                            f"{os.getcwd()}/{self.save_location}NIFTI_ROI")


class MatchedBrains:
    critical_parameters = []

    def __init__(self, brains, parameters):
        self.brains = brains
        self.parameters = parameters

    def test(self):
        print(2)


def compile_roi_stats(roiTempStore, roiResults):
    warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

    if config.noise_cutoff:
        noise_threshold = np.nanmean(roiTempStore[0, :])
        rois = roiTempStore[1:, :]
        below_threshold = np.count_nonzero(rois < noise_threshold, axis=1)
        roiResults[7, 1:-1] += below_threshold
        rois[rois < noise_threshold] = np.nan
        roiTempStore[1:, :] = rois

    axis = 1
    read_start = 0
    write_start = 0
    write_end = -1

    # First loop calculates summary stats for normal ROIs, second loop calculates stats for overall ROI
    for i in range(2):
        roiResults[0, write_start:write_end] = np.count_nonzero(~np.isnan(roiTempStore[read_start:, :]),
                                                                axis=axis)  # Count number of non-nan voxels
        roiResults[1, write_start:write_end] = np.nanmean(roiTempStore[read_start:, :], axis=axis)
        roiResults[2, write_start:write_end] = np.nanstd(roiTempStore[read_start:, :], axis=axis)
        roiResults[3, write_start:write_end] = Environment_Setup.conf_level_list[int(config.conf_level_number)][1] \
                                               * roiResults[2, write_start:write_end] \
                                               / np.sqrt(
            roiResults[0, write_start:write_end])  # 95% confidence interval calculation
        roiResults[4, write_start:write_end] = np.nanmedian(roiTempStore[read_start:, :], axis=axis)
        roiResults[5, write_start:write_end] = np.nanmin(roiTempStore[read_start:, :], axis=axis)
        roiResults[6, write_start:write_end] = np.nanmax(roiTempStore[read_start:, :], axis=axis)

        axis = None
        read_start = 1
        write_start = -1
        write_end = None

    roiResults[7, -1] = np.sum(roiResults[7, 1:-1])  # Calculate excluded voxels

    # Convert ROIs with no voxels from columns with NaNs to zeros
    for column, voxel_num in enumerate(roiResults[0]):
        if voxel_num == 0.0:
            for row in list(range(1, 6)):
                roiResults[row][column] = 0.0

    warnings.filterwarnings('default')  # Reactivate warnings

    return roiResults


def run_flirt_cost_function(fslfunc, ref, init, out_file, matrix_file, config):
    fslfunc.inputs.reference = ref
    fslfunc.inputs.args = f"-init {init}"  # args used as in_matrix_file method not working
    fslfunc.inputs.out_file = out_file
    fslfunc.inputs.out_matrix_file = matrix_file

    if config.verbose_cmd_line_args:
        print(fslfunc.cmdline)

    output = fslfunc.run()
    cost_func = float(re.search("[0-9]*\.[0-9]+", output.runtime.stdout)[0])

    # Clean up files
    os.remove(out_file)
    os.remove(matrix_file)

    return cost_func


def fsl_functions(obj, save_location, no_ext_brain, func, input, prefix, *argv):
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
        obj.brain = current_brain  # TODO comment this and how it effects other parts of the program

    elif func == 'BET':
        fslfunc.inputs.functional = True

    elif func == 'FLIRT':
        fslfunc.inputs.reference = argv[0]
        fslfunc.inputs.dof = config.dof
        current_mat = fslfunc.inputs.out_matrix_file = f'{save_location}{prefix}{no_ext_brain}.mat'

    elif func == 'ConvertXFM':
        if len(argv) > 0 and argv[0] == 'concat_xfm': # TODO: Double check if concatenated in correct direction
            fslfunc.inputs.in_file2 = obj.anat_brain_to_mni
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
        obj.file_list.extend([current_brain, current_mat])

    elif func == 'BET':
        obj.file_list.extend([current_brain, f"{save_location}{prefix}{no_ext_brain}_mask{suffix}"])

    else:
        obj.file_list.append(current_brain)

    if func == 'FLIRT':
        return current_mat

    return current_brain


def calculate_confidence_interval(data, alpha, roi=None):
    warnings.filterwarnings(action='ignore',
                            category=PendingDeprecationWarning)  # Silences a deprecation warning from bootstrapping library using outdated numpy matrix instead of numpy array

    if roi is None:
        data = data.flatten()
        values = np.array([x for x in data if str(x) != 'nan'])
    else:
        values = np.array([x for x in data[roi, :] if str(x) != 'nan'])

    results = bs.bootstrap(values, stat_func=bs_stats.mean, alpha=alpha, iteration_batch_size=10, num_threads=-1)
    conf_int = (results.upper_bound - results.lower_bound) / 2  # TODO: URGENT CHANGE THIS

    warnings.simplefilter(action='default', category=PendingDeprecationWarning)

    return results.value, conf_int


def verify_param_values():
    """Compare critical parameter choices to those in paramValues.csv. Exit with exception if discrepancy found."""
    from .paramparser import ParamParser

    table = [x.lower() for x in ParamParser.load_paramValues_file()][1:-1]

    for key in config.parameter_dict.keys():
        if key.lower() not in table:
            raise Exception(f'Key "{key}" not found in paramValues.csv. Check the Critical Parameters option '
                            f'in the Parsing menu (parameter_dict1 if not using the GUI) correctly match the '
                            f'paramValues.csv headers.')

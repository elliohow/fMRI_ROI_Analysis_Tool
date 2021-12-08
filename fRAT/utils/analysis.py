import itertools
import os
import re
import shutil
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
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans

from .utils import Utils

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
    label_array = []

    @classmethod
    def setup_analysis(cls, cfg, pool):
        """Set up environment and find files before running analysis."""
        global config
        config = cfg

        if config.verbose:
            print('\n--- Environment Setup ---')

        cls.setup_environment(config)

        if config.run_plotting:
            verify_param_values()

        cls.setup_save_location(config)

        # Extract labels from selected FSL atlas
        cls.roi_label_list()

        participant_list = Participant.setup_class(cls.base_directory, cls.save_location, pool)
        matched_brain_list = MatchedBrain.setup_class(participant_list)

        return participant_list, matched_brain_list

    @classmethod
    def setup_save_location(cls, config):
        cls.atlas_name = os.path.splitext(cls.atlas_label_list[int(config.atlas_number)][1])[0]

        if config.output_folder == 'DEFAULT':
            cls.save_location = f'{cls.atlas_name}_ROI_report/'
        else:
            cls.save_location = f'{config.output_folder}/'

        if config.verbose:
            print(f'Saving output in directory: {cls.save_location}\n'
                  f'Using the {cls.atlas_name} atlas.\n')

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(f"{cls.base_directory}/{cls.save_location}", delete_old=True)

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
                print(f'Finding subject directories in directory: {config.brain_file_loc}\n'
                      f'Looking for stat maps in directory: {config.stat_map_folder}\n')

        # Save copy of config_log.toml to retain settings. It is saved here as after changing directory it will be harder to find
        Utils.save_config(cls.base_directory, 'fRAT_config')

        try:
            os.chdir(cls.base_directory)
        except FileNotFoundError:
            raise FileNotFoundError('brain_file_loc in fRAT_config.toml is not a valid directory.')

    @classmethod
    def roi_label_list(cls):
        """Extract labels from specified FSL atlas XML file."""
        cls.atlas_path = f'{cls.fsl_path}/data/atlases/{cls.atlas_label_list[int(config.atlas_number)][0]}'
        cls.atlas_label_path = f'{cls.fsl_path}/data/atlases/{cls.atlas_label_list[int(config.atlas_number)][1]}'

        with open(cls.atlas_label_path) as fd:
            atlas_label_dict = xmltodict.parse(fd.read())

        cls.label_array = []
        cls.label_array.append('No ROI')

        for roiLabelLine in atlas_label_dict['atlas']['data']['label']:
            cls.label_array.append(roiLabelLine['#text'])

        cls.label_array.append('Overall')


class Participant:
    def __init__(self, participant, base_directory, save_location):
        self.participant_name = participant
        self.participant_path = f"{base_directory}/{participant}"
        self.save_location = f"{base_directory}/{save_location}{participant}/"
        self.anat_cost = 0
        self.brains = None
        self.file_list = []
        self.anat_brain = None
        self.anat_to_mni_mat = None
        self.anat_brain_no_ext = None
        self.fsl_segmentation = None

    def setup_participant(self, environment_globals, cfg):
        global config
        config = cfg

        self.find_fmri_files()  # Find fMRI files to save in self.brains

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(self.save_location)

        if config.anat_align:
            self.anat_setup(environment_globals['fsl_path'])

        # Initialise brain class for each file found
        for counter, brain in enumerate(self.brains):
            self.brains[counter] = Brain(f"{self.participant_path}/func/{brain}",
                                         self.participant_path,
                                         self.participant_name,
                                         self.save_location,
                                         self.anat_brain,
                                         self.anat_to_mni_mat,
                                         self.fsl_segmentation,
                                         environment_globals)

        return self

    @staticmethod
    def setup_class(base_directory, save_location, pool):
        _, participants = Utils.find_participant_dirs(os.getcwd())

        participant_list = set()
        for participant in participants:
            # Initialise participants
            participant_list.add(Participant(participant, base_directory, save_location))

        # Set arguments to pass to run_analysis function
        iterable = zip(participant_list, itertools.repeat("setup_participant"),
                       itertools.repeat(Environment_Setup.__dict__), itertools.repeat(config))

        if config.verbose and config.anat_align:
            print(f'\n--- Anatomical file alignment ---')

        # Setup each participant
        if config.multicore_processing:
            participant_list = set(pool.starmap(Utils.instance_method_handler, iterable))

        else:
            participant_list = set(itertools.starmap(Utils.instance_method_handler, iterable))

        return participant_list

    @classmethod
    def find_participant_dirs(cls):
        # Searches for folders that start with p
        participant_dirs = [direc for direc in glob("*") if re.search("^sub-[0-9]+", direc)]

        if len(participant_dirs) == 0:
            raise FileNotFoundError('Participant directories not found.')
        elif config.verbose:
            print(f'Found {len(participant_dirs)} participant folders.')

        return participant_dirs

    def run_analysis(self, pool):
        if config.verbose:
            print(f'\nAnalysing fMRI files for participant: {self.participant_name}\n')

        orig_brain_num = len(self.brains)
        self.brains = [brain for brain in self.brains if
                       brain.parameters]  # Remove any brains that are set to be ignored

        if config.verbose and orig_brain_num != len(self.brains):
            print(f'Ignoring {orig_brain_num - len(self.brains)}/{orig_brain_num} files.\n')

        brain_list = []
        if self.brains:  # self.brains will be empty if all files for this participant have been set to ignore

            # Set arguments to pass to run_analysis function
            iterable = zip(self.brains, itertools.repeat("run_analysis"), range(len(self.brains)),
                           itertools.repeat(len(self.brains)), itertools.repeat(config))

            if config.multicore_processing:
                brain_list = pool.starmap(Utils.instance_method_handler, iterable)
            else:
                brain_list = list(itertools.starmap(Utils.instance_method_handler, iterable))

            construct_combined_results(f'{self.save_location}')

        if config.anat_align:
            self.anat_file_cleanup()

        return brain_list

    def find_fmri_files(self):
        # Find all nifti and analyze files
        self.brains = Utils.find_files(f"{self.participant_path}/func", "hdr", "nii.gz", "nii")

        if len(self.brains) == 0:
            raise NameError("No files found.")

    def anat_setup(self, fsl_path):
        if config.verbose:
            print(f'Converting anatomical file to MNI space for participant: {self.participant_name}')

        anat = Utils.find_files(f'{self.participant_path}/anat/', "hdr", "nii.gz", "nii")

        if len(anat) > 1:
            raise FileExistsError('Multiple files found in anat folder.')
        else:
            anat = anat[0]

        self.anat_brain = f'{self.participant_path}/anat/{anat}'
        self.anat_brain_no_ext = anat.rsplit(".")[0]
        self.anat_to_mni_mat = fsl_functions(self, self.save_location, self.anat_brain_no_ext,
                                             'FLIRT', self.anat_brain, 'to_mni_from_',
                                             f'{fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')

        if config.grey_matter_segment:
            try:
                self.fsl_segmentation = glob(f"{self.participant_path}/fslfast/*_pve_1*")[0]
            except IndexError:
                self.fsl_segmentation = glob(f"{self.participant_path}/fslfast/*")[0]

    def calculate_anat_flirt_cost_function(self):
        fslfunc = fsl.FLIRT(in_file=self.anat_brain,
                            schedule=f'{Environment_Setup.fsl_path}/etc/flirtsch/measurecost1.sch',
                            terminal_output='allatonce', dof=config.dof)

        # Calculate MNI cost function value
        mni_cost = run_flirt_cost_function(fslfunc,
                                           f'{Environment_Setup.fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz',
                                           f'{self.save_location}Intermediate_files/to_mni_from_{self.anat_brain_no_ext}.mat',
                                           f'{self.save_location}Intermediate_files/{self.anat_brain_no_ext}_mni_redundant.nii.gz',
                                           f'{self.save_location}Intermediate_files/{self.anat_brain_no_ext}_mni_redundant.mat',
                                           config)

        d = {'Participant': [self.participant_name],
             'File': [self.anat_brain_no_ext],
             '(FLIRT to MNI) Cost function value': [mni_cost]}

        df = pd.DataFrame(data=d)

        return df

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
    def __init__(self, brain, participant_folder, participant_name, save_location,
                 anat_brain, anat_to_mni_mat, fsl_segmentation, environment_globals):
        self.brain = brain
        self.save_location = save_location
        self.anat_brain = anat_brain
        self.anat_to_mni_mat = anat_to_mni_mat
        self.fsl_segmentation = fsl_segmentation
        self.no_ext_brain = splitext(self.brain.split('/')[-1])[0]
        self.stat_brain = f"{participant_folder}/{config.stat_map_folder}{self.no_ext_brain}{config.stat_map_suffix}"
        self.roi_results = None
        self.roi_temp_store = None
        self.roi_stat_list = ""
        self.file_list = []
        self.participant_name = participant_name
        self.parameters = []
        self.noise_threshold = None
        self.lower_gaussian_threshold = None
        self.upper_gaussian_threshold = None

        # Copying class attributes here is a workaround for dill,
        # which can't access modified class attributes for imported modules.
        self._brain_directory = environment_globals['base_directory']
        self._fsl_path = environment_globals['fsl_path']
        self._atlas_label_path = environment_globals['atlas_label_path']
        self._atlas_name = environment_globals['atlas_name']
        self._labelArray = environment_globals['label_array']
        self.atlas_path = environment_globals['atlas_path']

    def fmri_get_additional_info(self, brain_number_current, brain_number_total, config):
        if config.verbose:
            print(f'Calculating cost function and mean displacement values for fMRI volume '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.no_ext_brain}')

        fslfunc = fsl.FLIRT(in_file=f'{self.save_location}Intermediate_files/bet_{self.no_ext_brain}.nii.gz',
                            schedule=f'{self._fsl_path}/etc/flirtsch/measurecost1.sch',
                            terminal_output='allatonce', dof=config.dof)

        anat_cost, mni_cost = None, None

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

        # Find absolute and relative mean displacement files
        suffixes = ['_abs_mean.rms', '_rel_mean.rms']
        displacement_vals = []
        for suffix in suffixes:
            with open(f"{self.save_location}Intermediate_files/motion_correction_files/mcf_"
                      f"{self.no_ext_brain}{suffix}", 'r') as file:
                displacement_vals.append(float(file.read().replace('\n', '')))

        d = {'Participant': [self.participant_name],
             'File': [self.no_ext_brain],
             '(FLIRT to MNI) Cost function value': mni_cost,
             '(MCFLIRT) Mean Absolute displacement': displacement_vals[0],
             '(MCFLIRT) Mean Relative displacement': displacement_vals[1]}

        if anat_cost:
            d['(FLIRT to anatomical) Cost function value'] = anat_cost
        if self.noise_threshold:
            d['Noise Threshold'] = self.noise_threshold
        if self.lower_gaussian_threshold:
            d['Lower Gaussian Outlier Threshold'] = self.lower_gaussian_threshold
        if self.upper_gaussian_threshold:
            d['Upper Gaussian Outlier Threshold'] = self.upper_gaussian_threshold

        df = pd.DataFrame(data=d)

        return df

    def run_analysis(self, brain_number_current, brain_number_total, cfg):
        global config
        config = cfg

        if config.verbose:
            print(f'Analysing fMRI volume {brain_number_current + 1}/{brain_number_total}: {self.no_ext_brain}')

        GM_bool = self.roi_flirt_transform(brain_number_current,
                                           brain_number_total)  # Convert MNI brain to native space

        self.roi_stats(GM_bool)  # Calculate and save statistics

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
        roi_temp_store = np.full([roiNum, idxMNI.shape[0]], np.nan)
        roi_results = np.full([8, roiNum + 1], np.nan)
        roi_results[7, 0:-1] = 0  # Change excluded voxels measure from NaN to 0

        return roi_temp_store, roi_results, idxMNI, idxBrain, roiList, roiNum

    @staticmethod
    def compile_voxel_values(roi_temp_store, roi_results, idxMNI, idxBrain, GM_bool):
        if config.grey_matter_segment:
            non_segmented_volume = roi_temp_store.copy()
        else:
            non_segmented_volume = None

        for counter, roi in enumerate(idxMNI):
            if config.grey_matter_segment:
                non_segmented_volume[int(roi), counter] = idxBrain[counter]

            if not config.grey_matter_segment or GM_bool[counter] == 1:
                roi_temp_store[int(roi), counter] = idxBrain[counter]

            else:
                roi_temp_store[0, counter] = idxBrain[counter]  # Assign to No ROI if voxel is excluded

                if int(roi) != 0:  # If ROI is not 'No ROI' add to excluded voxels list
                    roi_results[7, int(roi)] += 1

        return roi_temp_store, roi_results, non_segmented_volume

    def segmentation_to_fmri(self, anat_aligned_mat, fMRI_volume, brain_number_current, brain_number_total):
        if config.verbose:
            print(f'Aligning fslfast segmentation to fMRI volume {brain_number_current + 1}/{brain_number_total}.')

        prefix = 'fslfast_to_'
        interp = 'trilinear'

        # Save inverse of fMRI to anat
        inverse_mat = fsl_functions(self, self.save_location, self.no_ext_brain, 'ConvertXFM', anat_aligned_mat,
                                    'inverse_anat_to_')

        # Apply inverse of matrix to chosen segmentation to convert it into native space
        segmentation_to_fmri = fsl_functions(self, self.save_location, self.no_ext_brain, 'ApplyXFM',
                                             self.fsl_segmentation, prefix, inverse_mat, fMRI_volume, interp)

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

        # Motion correction
        fMRI_volume = fsl_functions(*pack_vars, 'MCFLIRT', self.brain, "mcf_")

        # Turn 4D scan into 3D
        fMRI_volume = fsl_functions(*pack_vars, 'MeanImage', fMRI_volume, "mean_")

        # Brain extraction
        fMRI_volume = fsl_functions(*pack_vars, 'BET', fMRI_volume, "bet_")

        if config.anat_align:
            if config.verbose:
                print(f'Aligning fMRI volume {brain_number_current + 1}/{brain_number_total} to anatomical volume.')

            # Align to anatomical
            fmri_to_anat_mat = fsl_functions(*pack_vars, 'FLIRT', fMRI_volume, "to_anat_from_", self.anat_brain)

            # Combine fMRI-anat and anat-mni matrices
            mat = fsl_functions(*pack_vars, 'ConvertXFM', fmri_to_anat_mat, 'combined_mat_', 'concat_xfm')

        else:
            # Align to MNI
            mat = fsl_functions(*pack_vars, 'FLIRT', fMRI_volume, "to_mni_from_",
                                f'{self._fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')

        # Get inverse of matrix
        inverse_mat = fsl_functions(*pack_vars, 'ConvertXFM', mat, 'inverse_combined_mat_')

        # Apply inverse of matrix to chosen atlas to convert it into native space
        fsl_functions(*pack_vars, 'ApplyXFM', self.atlas_path,
                      'mni_to_', inverse_mat, fMRI_volume,
                      'nearestneighbour')

        if config.grey_matter_segment:
            # Convert segmentation to fMRI native space
            segmentation_to_fmri = self.segmentation_to_fmri(fmri_to_anat_mat, fMRI_volume,
                                                             brain_number_current, brain_number_total)

            grey_matter_bool = self.find_gm_from_segment(segmentation_to_fmri)

            return grey_matter_bool

    def roi_stats(self, GM_bool):
        """Function which uses the output from the roi_flirt_transform function to collate the statistical information
        per ROI."""

        # Load brains and pre-initialise arrays
        roi_temp_store, roi_results, idxMNI, idxBrain, roiList, roiNum = self.roi_stats_setup()

        # Combine information from fMRI and MNI brains (both in native space) to assign an ROI to each voxel
        roi_temp_store, roi_results, non_segmented_volume = self.compile_voxel_values(roi_temp_store,
                                                                                      roi_results,
                                                                                      idxMNI,
                                                                                      idxBrain,
                                                                                      GM_bool)

        warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

        if non_segmented_volume is None:
            volumes = {'noROI': roi_temp_store}
        else:
            volumes = {'noROI': non_segmented_volume, 'nonGreyMatter': roi_temp_store}

        header, statmap_shape, save_location, stage = self.create_excluded_rois_volume(volumes)

        # Remove outliers from ROIs
        roi_temp_store, roi_results = self.outlier_detection(roi_results, roi_temp_store,
                                                             save_location, statmap_shape,
                                                             header, stage)

        # Compile ROI statistics
        roi_results = compile_roi_stats(roi_temp_store, roi_results, config)

        warnings.filterwarnings('default')  # Reactivate warnings

        # if config.bootstrap: # TODO: uncomment this when bootstrapping reimplemented
        #     roi_results = roi_stats_bootstrap(roi_temp_store, roi_results, roiNum, brain_number_current,
        #                                           brain_number_total)  # Bootstrapping

        roi_stats_save(roi_temp_store, roi_results, self._labelArray,
                       self.save_location, self.parameters, config)  # Save results

        self.roi_results = roi_results  # Retain variable for atlas_scale function
        self.roi_temp_store = roi_temp_store  # Retain variable for atlas_scale function

    def create_excluded_rois_volume(self, volumes):
        excluded_vox_save_location = f"{self.save_location}/Excluded_voxels/"
        Utils.check_and_make_dir(excluded_vox_save_location)

        statmap, header = Utils.load_brain(self.stat_brain)
        statmap_shape = statmap.shape

        stage = 1

        for file_name, volume in volumes.items():
            file_location = f"{excluded_vox_save_location}/ExcludedVoxStage{stage}_{self.no_ext_brain}_{file_name}.nii.gz"
            stage += 1

            create_no_roi_volume(
                volume,
                file_location,
                statmap_shape,
                header
            )

        return header, statmap_shape, excluded_vox_save_location, stage

    def outlier_detection(self, roi_results, roi_temp_store, save_location, statmap_shape, header, stage):

        if config.noise_cutoff:
            roi_results, roi_temp_store = self.noise_threshold_outlier_detection(
                header,
                roi_results,
                roi_temp_store,
                save_location,
                statmap_shape,
                stage
            )

            stage += 1

        if config.gaussian_outlier_detection:
            roi_results, roi_temp_store = self.gaussian_outlier_detection(
                header,
                roi_results,
                roi_temp_store,
                save_location,
                statmap_shape,
                stage
            )

            stage += 1

        return roi_temp_store, roi_results

    def gaussian_outlier_detection(self, header, roi_results, roi_temp_store, save_location, statmap_shape, stage):
        # Convert gaussian outlier contamination into confidence interval to retain for scipy norm.interval function
        confidence_interval = 1 - config.gaussian_outlier_contamination

        # Fit a gaussian to the data using EllipticEnvelope
        outliers = self.outlier_detection_using_gaussian(data=roi_temp_store[1:, :],
                                                         contamination=confidence_interval)
        outliers.sort()  # Sort to make it easier to find start and end

        outlier_bool_array = np.isin(roi_temp_store[1:, :], outliers)
        roi_results = self.calculate_number_of_outliers_per_roi(outlier_bool_array, roi_results)
        roi_temp_store = self.remove_outliers(outlier_bool_array, roi_temp_store)

        create_no_roi_volume(roi_temp_store,
                             f"{save_location}ExcludedVoxStage{stage}_{self.no_ext_brain}_gaussianThreshOutliers.nii.gz",
                             statmap_shape,
                             header)

        return roi_results, roi_temp_store

    def noise_threshold_outlier_detection(self, header, roi_results, roi_temp_store, save_location, statmap_shape,
                                          stage):
        # Calculate noise threshold
        self.noise_threshold = np.true_divide(np.nansum(roi_temp_store[0, :]),
                                              np.count_nonzero(roi_temp_store[0, :]))

        outlier_bool_array = roi_temp_store[1:, :] < self.noise_threshold
        roi_results = self.calculate_number_of_outliers_per_roi(outlier_bool_array, roi_results)
        roi_temp_store = self.remove_outliers(outlier_bool_array, roi_temp_store)

        create_no_roi_volume(roi_temp_store,
                             f"{save_location}ExcludedVoxStage{stage}_{self.no_ext_brain}_noiseThreshOutliers.nii.gz",
                             statmap_shape,
                             header)

        return roi_results, roi_temp_store

    def outlier_detection_using_gaussian(self, data, contamination):
        values = data.copy()
        values = values[~np.isnan(values)]

        prediction = norm(*norm.fit(values))
        gaussian_lims = prediction.interval(contamination)

        if config.gaussian_outlier_location == 'below gaussian':
            outliers = [value for value in values if value < gaussian_lims[0]]
            self.lower_gaussian_threshold = gaussian_lims[0]

        elif config.gaussian_outlier_location == 'above gaussian':
            outliers = [value for value in values if value > gaussian_lims[1]]
            self.upper_gaussian_threshold = gaussian_lims[1]

        elif config.gaussian_outlier_location == 'both':
            outliers = [value for value in values if value < gaussian_lims[0] or value > gaussian_lims[1]]
            self.lower_gaussian_threshold = gaussian_lims[0]
            self.upper_gaussian_threshold = gaussian_lims[1]

        else:
            raise Exception('gaussian_outlier_location not valid')

        return outliers

    def calculate_number_of_outliers_per_roi(self, outlier_bool_array, roi_results):
        # Count how many voxels are below noise threshold for each ROI
        below_threshold = np.count_nonzero(outlier_bool_array, axis=1)
        # Assign voxels below threshold to excluded voxels column
        roi_results[7, 1:-1] += below_threshold

        return roi_results

    def remove_outliers(self, outlier_bool_array, roi_temp_store):
        # Convert from bool to float64 to be able to assign values
        outlier_float_array = np.max(outlier_bool_array, axis=0).astype('float64')

        # Replace False with nans and True with statistic value
        outlier_float_array[outlier_float_array == 0.0] = np.nan

        # Use the transpose of roi_temp_store and outlier bool array to make sure output is ordered by columns not rows
        outlier_float_array[outlier_float_array == 1.0] = roi_temp_store[1:, :].T[outlier_bool_array.T]

        # Set any outlier ROI values to nan
        roi_temp_store[1:, :][outlier_bool_array] = np.nan

        # Stack no ROI values on top of array containing outlier values, and then use nan max to condense into one array
        roi_temp_store[0, :] = np.nanmax(np.vstack((outlier_float_array, roi_temp_store[0, :])), axis=0)

        return roi_temp_store


class MatchedBrain:
    label_array = []
    save_location = None

    def __init__(self, brains, parameters):
        self.brains = brains
        self.parameters = "_".join([f"{param_name}{param_val}" for param_name, param_val
                                    in zip(config.parameter_dict2, parameters)])
        self.overall_results = []
        self.raw_results = []

        # Copying class attributes here is a workaround for dill, which can't access class attributes.
        self.label_array = self.label_array
        self.save_location = self.save_location

    @classmethod
    def setup_class(cls, participant_list):
        matched_brains = cls.find_shared_params(participant_list)  # Find brains which share parameter combinations

        cls.label_array = Environment_Setup.label_array
        cls.save_location = f"{Environment_Setup.save_location}Overall/"
        Utils.check_and_make_dir(cls.save_location)

        matched_brain_list = set()
        for param_combination in matched_brains:
            # Initialise participants
            matched_brain_list.add(MatchedBrain(matched_brains[param_combination], param_combination))

        cls.assign_parameters_to_brains(matched_brain_list, participant_list)

        return matched_brain_list

    @classmethod
    def find_shared_params(cls, participant_list):
        table = load_paramValues_file()

        ignore_column_loc, critical_column_locs = cls.find_column_locs(table)

        matched_brains = dict()
        for index, row in table.iterrows():
            if ignore_column_loc and str(row[ignore_column_loc]).strip().lower() in (
                    'yes', 'y', 'true'):  # If column is set to ignore then do not include it in analysis
                continue

            elif tuple(row[critical_column_locs]) not in matched_brains.keys():
                matched_brains[tuple(row[critical_column_locs])] = dict()

            elif row['participant'] in matched_brains[tuple(row[critical_column_locs])].keys():
                raise FileExistsError(f'Multiple instances of participant {row["participant"]} found for parameter '
                                      f'combination {tuple(row[critical_column_locs])}. '
                                      f'This is not currently supported.')

            participant, brain = cls.find_brain_object(row, participant_list)
            matched_brains[tuple(row[critical_column_locs])][participant] = brain

        return matched_brains

    @classmethod
    def find_column_locs(cls, table):
        table.columns = [x.lower() for x in table.columns]  # Convert to lower case for comparison to key later

        ignore_column_loc = next((counter for counter, column in enumerate(table.columns) if "ignore file" in column),
                                 False)

        critical_column_locs = set()
        for key in config.parameter_dict:
            column_loc = next((counter for counter, column in enumerate(table.columns) if key.lower() == column), False)

            if column_loc:
                critical_column_locs.add(column_loc)
            else:
                raise Exception(f'Key "{key}" not found in paramValues.csv. Check the Critical Parameters option '
                                f'in the Parsing menu (parameter_dict1 if not using the GUI) correctly match the '
                                f'paramValues.csv headers.')

        return ignore_column_loc, critical_column_locs

    @staticmethod
    def find_brain_object(row, participant_list):
        participant = next(participant for participant in participant_list
                           if participant.participant_name == row['participant'])

        brain = next(brain for brain in participant.brains
                     if brain.no_ext_brain == row['file name'])

        return participant.participant_name, brain.no_ext_brain

    def compile_results(self, config):
        if config.verbose:
            print(f'Combining results for parameter combination: {self.parameters}')

        for result in self.overall_results:
            result[0:-1, :] = 0  # Reset roi_results and only retain excluded voxels

        # Collapse overall results and calculate total excluded voxels
        self.overall_results = np.sum(self.overall_results, axis=0)

        # Combine raw results
        self.raw_results = np.concatenate(self.raw_results, axis=1)

        # Compile ROI statistics
        self.overall_results = compile_roi_stats(self.raw_results, self.overall_results, config)

        # Save results
        roi_stats_save(self.raw_results, self.overall_results, self.label_array,
                       self.save_location, self.parameters, config)

        return self

    @classmethod
    def assign_parameters_to_brains(cls, matched_brains, participant_list):
        brain_list = []
        for participant in participant_list:
            for brain in participant.brains:
                brain_list.append(brain)

        for parameter_comb in matched_brains:
            for brain in brain_list:
                try:
                    if parameter_comb.brains[brain.participant_name] == brain.no_ext_brain:
                        brain.parameters = parameter_comb.parameters

                except KeyError:
                    pass

    def atlas_scale(self, max_roi_stat, brain_number_current, brain_number_total, statistic_num, atlas_path, config):
        """Produces up to three scaled NIFTI files. Within brains, between brains (based on rois), between brains
        (based on the highest seen value of all brains and rois)."""
        if config.verbose and max(max_roi_stat) != 0.0:
            print(f'Creating {config.statistic_options[statistic_num]} NIFTI_ROI file for parameter combination '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.parameters}.')

        elif config.verbose and \
                config.statistic_options[statistic_num] == 'Excluded_voxels_amount' and max(max_roi_stat) == 0.0:
            print(f'Not creating {config.statistic_options[statistic_num]} NIFTI_ROI file for parameter combination '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.parameters} as no voxels have been excluded.')

            return

        roi_scaled_stat = [(y / x) * 100 for x, y in zip(max_roi_stat, self.overall_results[statistic_num, :])]
        # Find maximum statistic value (excluding No ROI and overall category)
        global_scaled_stat = [(y / max(max_roi_stat[1:-1])) * 100 for y in self.overall_results[statistic_num, :]]

        atlas = nib.load(atlas_path)
        atlas = atlas.get_fdata()

        unscaled_stat, within_roi_stat, mixed_roi_stat = self.group_roi_stats(atlas, global_scaled_stat,
                                                                              roi_scaled_stat, statistic_num)

        # Convert atlas to NIFTI and save it
        scale_stats = [
            (unscaled_stat,
             f"{self.parameters}_{config.statistic_options[statistic_num]}.nii.gz"),
            (within_roi_stat,
             f"{self.parameters}_{config.statistic_options[statistic_num]}_within_roi_scaled.nii.gz"),
            (mixed_roi_stat,
             f"{self.parameters}_{config.statistic_options[statistic_num]}_mixed_roi_scaled.nii.gz")
        ]

        for scale_stat in scale_stats:
            scaled_brain = nib.Nifti1Image(scale_stat[0], np.eye(4))
            scaled_brain.to_filename(f"{self.save_location}NIFTI_ROI/{scale_stat[1]}")

    def group_roi_stats(self, atlas, global_scaled_stat, roi_scaled_stat, statistic_num):
        # Iterate through each voxel in the atlas
        atlas = atlas.astype(int)

        # Assign stat values for each ROI all at once
        unscaled_stat = self.overall_results[statistic_num, atlas]
        within_roi_stat = np.array(roi_scaled_stat)[atlas]
        mixed_roi_stat = np.array(global_scaled_stat)[atlas]

        # Make ROI group 0 (No ROI) nan so it does not effect colourmap when viewing in fsleyes
        unscaled_stat[atlas == 0] = np.nan
        within_roi_stat[atlas == 0] = np.nan
        mixed_roi_stat[atlas == 0] = np.nan

        return unscaled_stat, within_roi_stat, mixed_roi_stat


def create_no_roi_volume(volume, save_location, statmap_shape, header):
    data = volume[0, :].copy()
    data = data.reshape(statmap_shape)

    brain = nib.Nifti1Pair(data, None, header)
    nib.save(brain, save_location)


def compile_roi_stats(roi_temp_store, roi_results, config):
    warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

    axis = 1
    read_start = 0
    write_start = 0
    write_end = -1

    # First loop calculates summary stats for normal ROIs, second loop calculates stats for overall ROI
    for i in range(2):
        roi_results[0, write_start:write_end] = np.count_nonzero(~np.isnan(roi_temp_store[read_start:, :]),
                                                                 axis=axis)  # Count number of non-nan voxels
        roi_results[1, write_start:write_end] = np.nanmean(roi_temp_store[read_start:, :], axis=axis)
        roi_results[2, write_start:write_end] = np.nanstd(roi_temp_store[read_start:, :], axis=axis)
        # Confidence interval calculation
        roi_results[3, write_start:write_end] = Environment_Setup.conf_level_list[int(config.conf_level_number)][1] \
                                                * roi_results[2, write_start:write_end] \
                                                / np.sqrt(roi_results[0, write_start:write_end])
        roi_results[4, write_start:write_end] = np.nanmedian(roi_temp_store[read_start:, :], axis=axis)
        roi_results[5, write_start:write_end] = np.nanmin(roi_temp_store[read_start:, :], axis=axis)
        roi_results[6, write_start:write_end] = np.nanmax(roi_temp_store[read_start:, :], axis=axis)

        axis = None
        read_start = 1
        write_start = -1
        write_end = None

    roi_results[7, -1] = np.sum(roi_results[7, 1:-1])  # Calculate excluded voxels

    # Convert ROIs with no voxels from columns with NaNs to zeros
    for column, voxel_num in enumerate(roi_results[0]):
        if voxel_num == 0.0:
            for row in list(range(1, 6)):
                roi_results[row][column] = 0.0

    warnings.filterwarnings('default')  # Reactivate warnings

    return roi_results


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
    current_mat = None
    current_brain, fslfunc, suffix = fsl_functions_setup(func, input, no_ext_brain, prefix, save_location)

    # Arguments dependent on FSL function used
    if func == 'MCFLIRT':
        fslfunc.inputs.save_rms = True

    elif func == 'BET':
        fslfunc.inputs.functional = True

    elif func == 'FLIRT':
        fslfunc.inputs.reference = argv[0]
        fslfunc.inputs.dof = config.dof
        current_mat = fslfunc.inputs.out_matrix_file = f'{save_location}{prefix}{no_ext_brain}.mat'

    elif func == 'ConvertXFM':
        if len(argv) > 0 and argv[0] == 'concat_xfm':
            fslfunc.inputs.in_file2 = obj.anat_to_mni_mat
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

    fsl_function_file_handle(current_brain, current_mat, func, no_ext_brain, obj, prefix, save_location, suffix)

    if func == 'FLIRT':
        return current_mat

    return current_brain


def fsl_function_file_handle(current_brain, current_mat, func, no_ext_brain, obj, prefix, save_location, suffix):
    if func in ('FLIRT', 'ApplyXFM'):
        obj.file_list.extend([current_brain, current_mat])
    elif func == 'BET':
        obj.file_list.extend([current_brain, f"{save_location}{prefix}{no_ext_brain}_mask{suffix}"])
    elif func == 'MCFLIRT':
        # Find all the motion correction files that are not the actual brain volume
        mc_files = [direc for direc in os.listdir(f"{save_location}motion_correction_files/")
                    if re.search(no_ext_brain, direc) and not re.search(f"^{prefix}{no_ext_brain}{suffix}$", direc)]

        # Remove .nii.gz from middle of string
        for file in mc_files:
            os.rename(f"{save_location}motion_correction_files/{file}",
                      f"{save_location}motion_correction_files/{file.replace(suffix, '')}")

    else:
        obj.file_list.append(current_brain)


def fsl_functions_setup(func, input, no_ext_brain, prefix, save_location):
    fslfunc = getattr(fsl, func)()
    fslfunc.inputs.in_file = input
    fslfunc.inputs.output_type = 'NIFTI_GZ'

    # Standard variables that may be changed for specific FSL functions
    suffix = '.nii.gz'
    current_brain = fslfunc.inputs.out_file = f"{save_location}{prefix}{no_ext_brain}{suffix}"

    if func == 'ConvertXFM':
        suffix = '.mat'
    elif func == 'MCFLIRT':
        Utils.check_and_make_dir(f"{save_location}motion_correction_files/")
        current_brain = fslfunc.inputs.out_file = f"{save_location}motion_correction_files/{prefix}{no_ext_brain}{suffix}"

    return current_brain, fslfunc, suffix


def calculate_confidence_interval(data, alpha, roi=None):
    warnings.filterwarnings(action='ignore',
                            category=PendingDeprecationWarning)  # Silences a deprecation warning from bootstrapping library using outdated numpy matrix instead of numpy array

    if roi is None:
        data = data.flatten()
        values = np.array([x for x in data if str(x) != 'nan'])
    else:
        values = np.array([x for x in data[roi, :] if str(x) != 'nan'])

    results = bs.bootstrap(values, stat_func=bs_stats.mean, alpha=alpha, iteration_batch_size=10, num_threads=-1)
    conf_int = (results.upper_bound - results.lower_bound) / 2
    # TODO: Change this so it returns asymmetric bootstrap and implement bootstrapping for pooled analysis

    warnings.simplefilter(action='default', category=PendingDeprecationWarning)

    return results.value, conf_int


def roi_stats_bootstrap(roi_temp_store, roi_results, roiNum, brain_number_current, brain_number_total):
    for counter, roi in enumerate(list(range(0, roiNum + 1))):
        if config.verbose:
            print(f"  - Bootstrapping ROI {counter + 1}/{roiNum + 1} "
                  f"for fMRI volume {brain_number_current + 1}/{brain_number_total}.")

        if counter < roiNum:
            roi_results[1, roi], roi_results[3, roi] = calculate_confidence_interval(roi_temp_store,
                                                                                     config.bootstrap_alpha,
                                                                                     roi=roi)
        else:
            # Calculate overall statistics
            roi_results[1, -1], roi_results[3, -1] = calculate_confidence_interval(roi_temp_store[1:, :],
                                                                                   config.bootstrap_alpha)
    return roi_results


def roi_stats_save(roi_temp_store, roi_results, labelArray, save_location, no_ext_brain, config):
    headers = ['Voxels', 'Mean', 'Std_dev',
               f'Conf_Int_{Environment_Setup.conf_level_list[int(config.conf_level_number)][0]}',
               'Median', 'Min', 'Max', 'Excluded_Voxels']

    # Reorganise matrix to later remove nan rows
    roi_temp_store = roi_temp_store.transpose()
    i = np.arange(roi_temp_store.shape[1])
    # Find indices of nans and put them at the end of each column
    a = np.isnan(roi_temp_store).argsort(0, kind='mergesort')
    # Reorganise matrix with nans at end
    roi_temp_store[:] = roi_temp_store[a, i]

    # Save results as dataframe
    results = pd.DataFrame(data=roi_results,
                           index=headers,
                           columns=labelArray)
    raw_results = pd.DataFrame(data=roi_temp_store,
                               columns=labelArray[:-1])

    # Remove the required rows from the dataframe
    raw_results = raw_results.drop(raw_results.columns[0], axis=1)

    # Remove rows where all columns have NaNs (essential to keep file size down)
    raw_results = raw_results.dropna(axis=0, how='all')

    # Convert to dict and get rid of row numbers to significantly decrease file size
    roidict = Utils.dataframe_to_dict(raw_results)

    summary_results_path = f"{save_location}Summarised_results/"
    Utils.check_and_make_dir(summary_results_path)

    with open(f"{summary_results_path}{no_ext_brain}.json", 'w') as file:
        json.dump(results.to_dict(), file, indent=2)

    raw_results_path = f"{save_location}Raw_results/"
    Utils.check_and_make_dir(raw_results_path)

    with open(f"{raw_results_path}{no_ext_brain}_raw.json", 'w') as file:
        json.dump(roidict, file, indent=2)


def verify_param_values():
    """Compare critical parameter choices to those in paramValues.csv. Exit with exception if discrepancy found."""
    table = [x.lower() for x in load_paramValues_file()][1:-1]

    for key in config.parameter_dict.keys():
        if key.lower() not in table:
            raise Exception(f'Key "{key}" not found in paramValues.csv. Check the Critical Parameters option '
                            f'in the Parsing menu (parameter_dict1 if not using the GUI) correctly match the '
                            f'paramValues.csv headers.')


def load_paramValues_file():
    if os.path.isfile(f"{os.getcwd()}/paramValues.csv"):
        table = pd.read_csv("paramValues.csv")  # Load param table
    else:
        try:
            table = pd.read_csv(f"copy_paramValues.csv")  # Load param table
        except FileNotFoundError:
            raise Exception('Make sure a copy of paramValues.csv is in the chosen folder.')

    return table


def construct_combined_results(directory):
    combined_dataframe = pd.DataFrame()
    json_file_list = [os.path.basename(f) for f in glob(f"{directory}/Summarised_results/*.json")]

    for jsn in json_file_list:
        if jsn == 'combined_results.json':
            continue

        # Splits a file name. For example from hb1_ip2.json into [1, 2]
        parameters = list(filter(None, re.split(f'_|.json|{config.parameter_dict2}', jsn)))

        current_dataframe = pd.read_json(f"{directory}/Summarised_results/{jsn}")
        current_dataframe = current_dataframe.transpose()

        for counter, parameter_name in enumerate(config.parameter_dict):
            current_dataframe[parameter_name] = parameters[counter]  # Add parameter columns

        current_dataframe['File_name'] = jsn.split('.')[0]

        if combined_dataframe.empty:
            combined_dataframe = current_dataframe

        else:
            combined_dataframe = combined_dataframe.append(current_dataframe, sort=True)

    # Save combined results
    combined_dataframe = combined_dataframe.reset_index()
    combined_dataframe.to_json(f"{directory}/Summarised_results/combined_results.json", orient='records', indent=2)


def json_search():
    if len(json_file_list) == 0:
        raise NameError('Folder selection error. Could not find json files in the "Summarised_results" directory.')
    else:
        return json_file_list

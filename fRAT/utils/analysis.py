import copy
import itertools
import logging
import os
import pathlib
import re
import shutil
import warnings
from glob import glob
from random import choice

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import nibabel as nib
import numpy as np
import pandas as pd
import simplejson as json
import xmltodict
from nipype.interfaces import fsl
from nipype.interfaces.fsl import maths
from scipy.stats import norm

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
    def setup_analysis(cls, cfg, config_path, config_filename, pool):
        """Set up environment and find files before running analysis."""
        global config
        config = cfg

        if config.verbose:
            print('\n--- Environment Setup ---')

        cls.setup_environment(config_path, config_filename)

        if config.run_plotting:
            verify_param_values()

        cls.setup_save_location()

        # Extract labels from selected FSL atlas
        cls.atlas_path = f'{cls.fsl_path}/data/atlases/{cls.atlas_label_list[int(config.atlas_number)][0]}'
        cls.atlas_label_path = f'{cls.fsl_path}/data/atlases/{cls.atlas_label_list[int(config.atlas_number)][1]}'

        cls.roi_label_list()

        participant_list = Participant.setup_class(cls.base_directory, cls.save_location, pool)
        matched_brain_list = MatchedBrain.setup_class(participant_list)

        return participant_list, matched_brain_list

    @classmethod
    def setup_save_location(cls):
        cls.atlas_name = os.path.splitext(cls.atlas_label_list[int(config.atlas_number)][1])[0]

        if config.output_folder == 'DEFAULT':
            cls.save_location = f'{cls.atlas_name}_ROI_report/'
        else:
            cls.save_location = f'{config.output_folder}/'

        if config.verbose:
            print(f'Saving output in directory: {cls.save_location}\n'
                  f'Using the {cls.atlas_name} atlas.\n'
                  f'Using parameter file: {config.parameter_file}.\n')

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(f"{cls.base_directory}/{cls.save_location}", delete_old=True)

        # Move config file to analysis folder
        Utils.move_file('analysis_log.toml', cls.base_directory, cls.save_location)

        if config.stat_map_folder == '':
            print('Statistical map folder to use in subject\'s statmap folders not specified.'
                  '\nIf only one folder is found in the statmap directory, this folder will be selected. '
                  '\nMake sure this folder contains the same kind of statistical map for each participant.'
                  '\nIn future consider filling in the statistical map folder field '
                  'as this information will then be added into the analysis_log.toml file for future reference.\n')
        elif config.verbose:
            print(f'Searching for statmaps in directory: statmaps/{config.stat_map_folder}/\n')

    @classmethod
    def setup_environment(cls, config_path, config_filename):
        try:
            cls.fsl_path = os.environ['FSLDIR']
        except (KeyError, OSError) as e:
            raise Exception('FSL environment variable not set.')

        if config.brain_file_loc in ("", " "):
            print('Select the directory of the raw MRI/fMRI brains.')
            cls.base_directory = Utils.file_browser(title='Select the directory of the raw MRI/fMRI brains')

        else:
            cls.base_directory = config.brain_file_loc

            if config.verbose:
                print(f'Finding subject directories in directory: {config.brain_file_loc}\n')

        # Save copy of analysis_log.toml to retain settings. It is saved here as after changing directory it will be harder to find
        Utils.save_config(cls.base_directory, config_path, config_filename,
                          new_config_name='analysis_log',
                          relevant_sections=['General', 'Analysis', 'Parsing'])

        try:
            os.chdir(cls.base_directory)

        except FileNotFoundError:
            raise FileNotFoundError('Chosen base directory of subjects is not a valid directory path.')

    @classmethod
    def roi_label_list(cls):
        """Extract labels from specified FSL atlas XML file."""
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
        self.anat_head = None
        self.anat_brain = None
        self.anat_to_mni_mat = None
        self.anat_brain_no_ext = None
        self.grey_matter_segmentation = None
        self.white_matter_segmentation = None
        self.all_files_ignored = False
        self.statmap_folder = ''

        self.find_statmap_folder()

    def find_statmap_folder(self):
        if config.stat_map_folder:
            self.statmap_folder = config.stat_map_folder

        else:
            statmap_folders = glob(os.path.join(f"{self.participant_path}/statmaps", '*'))

            if len(statmap_folders) > 1:
                raise FileExistsError(f"No statistical map folder specified, however multiple statistical map folders "
                                      f"found in {self.participant_name}'s statmap folder.")
            elif len(statmap_folders) == 0:
                raise FileNotFoundError(f"No statistical maps found in {self.participant_name}'s statmap folder.")

            else:
                self.statmap_folder = os.path.split(statmap_folders[0])[1]

                if config.verbose:
                    print(f'Searching for {self.participant_name} statmaps in directory: statmaps/{self.statmap_folder}/\n')

    def setup_participant(self, environment_globals, cfg):
        global config
        config = cfg

        self.find_fmri_files()  # Find fMRI files to save in self.brains

        # Make folder to save ROI_report if not already created
        Utils.check_and_make_dir(self.save_location)

        self.anat_setup(environment_globals['fsl_path'])

        # Initialise brain class for each file found
        for counter, brain in enumerate(self.brains):
            self.brains[counter] = Brain(f"{self.participant_path}/{config.input_folder_name}/{brain}",
                                         self.participant_path,
                                         self.participant_name,
                                         self.save_location,
                                         self.anat_head,
                                         self.anat_brain,
                                         self.anat_to_mni_mat,
                                         self.grey_matter_segmentation,
                                         self.white_matter_segmentation,
                                         self.statmap_folder,
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

        if config.verbose:
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

            construct_combined_results(self.save_location, analysis_type='participant')

            self.anat_file_cleanup()

        else:
            self.all_files_ignored = True
            shutil.rmtree(self.save_location)

        return brain_list

    def find_fmri_files(self):
        # Find all nifti and analyze files
        self.brains = Utils.find_files(f"{self.participant_path}/{config.input_folder_name}", "hdr", "nii.gz", "nii")

        if len(self.brains) == 0:
            raise NameError("No files found.")

    @staticmethod
    def anat_file_check(files, filetype):
        if len(files) > 1:
            raise FileExistsError(f'More than one {filetype} images found in the anat folder.')
        elif len(files) == 0:
            raise FileExistsError(f'No {filetype} images found in the anat folder.')
        else:
            return files[0]

    def anat_setup(self, fsl_path):
        if config.verbose:
            print(f'Converting anatomical file to MNI space for participant: {self.participant_name}')

        anat = Utils.find_files(f'{self.participant_path}/anat/', "hdr", "nii.gz", "nii")

        if config.anat_align_cost_function == 'BBR' and len(anat) < 2:
            raise FileExistsError('When using the BBR cost function for the fMRI to structural registration, the anat '
                                  'folder should contain both the whole head and brain extracted images labelled in '
                                  'the filename with head and brain respectively e.g. "MPRAGE_head", "MPRAGE_brain".')
        elif len(anat) > 2:
            raise FileExistsError(
                'The maximum number of files NIFTI in the anat folder should be two: a whole head and a brain '
                'extracted image, where the brain extracted image has "brain" in the filename e.g. '
                '"MPRAGE_brain". Remove any json or mask files from this folder.'
                '\nNOTE: Wholehead image is only required when '
                'aligning fMRI volume to anatomical volume with the BBR cost function.')

        if len(anat) == 1:
            brain = anat[0]
        else:
            brain = [file for file in anat if 'brain' in file]
            brain = self.anat_file_check(brain, 'brain')

        if config.anat_align_cost_function == 'BBR':
            head = [file for file in anat if 'brain' not in file]
            head = self.anat_file_check(head, 'whole head')

            self.anat_head = f'{self.participant_path}/anat/{head}'

        self.anat_brain = f'{self.participant_path}/anat/{brain}'
        self.anat_brain_no_ext = brain.rsplit(".")[0]
        self.anat_to_mni_mat = fsl_functions(self, self.save_location, self.anat_brain_no_ext,
                                             'FLIRT', self.anat_brain, 'to_mni_from_',
                                             f'{fsl_path}/data/standard/MNI152_T1_1mm_brain.nii.gz')

        if config.grey_matter_segment or config.anat_align_cost_function == 'BBR':
            try:
                self.find_fslfast_files()

            except IndexError:
                if config.run_fsl_fast == 'Run if files not found':
                    if config.verbose:
                        phrases = ['get a cup of coffee',
                                   'respond to those reviewer comments',
                                   'have a good clear out of your inbox',
                                   'read some XKCD comics',
                                   'contemplate the meaning of existence',
                                   'consider writing a short poem',
                                   'listen to Echoes by Pink Floyd',
                                   'mark one of those essays you\'ve been putting off',
                                   'paint the next abstract masterpiece',
                                   'listen to Michigan by Sufjan Stevens',
                                   'get up and have a stretch',
                                   'follow @elliohow on twitter',
                                   'finally learn the Brodmann areas',
                                   'read the Tidy Data paper by Hadley Wickham']

                        print(f'Participant {self.participant_name} missing FSL FAST files. '
                              f'Running FSL FAST (maybe {choice(phrases)}, this may take a while).')

                    # Need to change directory here to get around error caused by nipype trying to find fslfast output
                    # in current working directory
                    orig_direc = os.getcwd()
                    Utils.check_and_make_dir(f'{self.participant_path}/fslfast/')
                    os.chdir(f'{self.participant_path}/fslfast/')

                    fsl_functions(self, f'{self.participant_path}/fslfast/{self.anat_brain_no_ext}',
                                  '', 'FAST', self.anat_brain, '')

                    os.chdir(orig_direc)

                    self.find_fslfast_files()

                else:
                    raise FileNotFoundError(f'fslfast directory for {self.participant_name} does not contain all files'
                                            f'output by FAST.')

    def find_fslfast_files(self):
        self.grey_matter_segmentation = glob(f"{self.participant_path}/fslfast/*_pve_1*")[0]
        self.white_matter_segmentation = glob(f"{self.participant_path}/fslfast/*_pve_2*")[0]

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
                 anat_head, anat_brain, anat_to_mni_mat, grey_matter_segmentation, white_matter_segmentation,
                 statmap_folder,
                 environment_globals):
        self.brain = brain
        self.save_location = save_location
        self.anat_head = anat_head
        self.anat_brain = anat_brain
        self.anat_to_mni_mat = anat_to_mni_mat
        self.grey_matter_segmentation = grey_matter_segmentation
        self.white_matter_segmentation = white_matter_segmentation
        self.no_ext_brain = Utils.strip_ext(self.brain.split('/')[-1])
        self.stat_brain = f"{participant_folder}/statmaps/{statmap_folder}/{self.no_ext_brain}{config.stat_map_suffix}"
        self.mni_brain = f"{self.save_location}mni_to_{self.no_ext_brain}.nii.gz"
        self.roi_results = None
        self.roi_temp_store = None
        self.roi_stat_list = ""
        self.file_list = []
        self.participant_name = participant_name
        self.parameters = []
        self.noise_threshold = None
        self.lower_gaussian_threshold = None
        self.upper_gaussian_threshold = None
        self.session_number = 0

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
            print(f'Calculating cost function and mean displacement values for volume '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.no_ext_brain}')

        fslfunc = fsl.FLIRT(
            in_file=f'{self.save_location}Intermediate_files/{self.no_ext_brain}/bet_{self.no_ext_brain}.nii.gz',
            schedule=f'{self._fsl_path}/etc/flirtsch/measurecost1.sch',
            terminal_output='allatonce', dof=config.dof)

        anat_cost, mni_cost = None, None

        # Calculate anatomical cost function value
        wmseg = None
        if config.anat_align_cost_function == 'BBR':
            wmseg = f'{self.save_location}Intermediate_files/{self.no_ext_brain}/to_anat_from_{self.no_ext_brain}_fast_wmseg.nii.gz'

        anat_cost = run_flirt_cost_function(fslfunc,
                                            self.anat_brain,
                                            f'{self.save_location}Intermediate_files/{self.no_ext_brain}/to_anat_from_{self.no_ext_brain}.mat',
                                            f'{self.save_location}Intermediate_files/{self.no_ext_brain}_anat_redundant.nii.gz',
                                            f'{self.save_location}Intermediate_files/{self.no_ext_brain}_anat_redundant.mat',
                                            config, wmseg=wmseg)

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

        logging.getLogger('nipype.interface').setLevel(0)  # Suppress nipype interface terminal output

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
            Utils.check_and_make_dir(f"{self.save_location}/Intermediate_files/{self.no_ext_brain}")

            for file in self.file_list:
                file = file.replace(self.save_location, "")  # Remove folder from start of file name
                Utils.move_file(file, self.save_location,
                                f"{self.save_location}/Intermediate_files/{self.no_ext_brain}")

        self.file_list = []

        redundant_files = glob(f'{self.save_location}/white_matter_thresholded_{self.no_ext_brain}.nii.gz') \
                          + glob(f'{self.save_location}/{self.no_ext_brain}_init.mat')

        for file in redundant_files:
            os.remove(file)

    def roi_stats_setup(self):
        # Load original brain (with statistical map)
        stat_brain = nib.load(self.stat_brain)
        # Load atlas brain (which has been converted into native space)
        mni_brain = nib.load(self.mni_brain)

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

        return roi_temp_store, roi_results, idxMNI, idxBrain

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
                                             self.grey_matter_segmentation, prefix, inverse_mat, fMRI_volume, interp)

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

        if config.verbose:
            print(f'Aligning fMRI volume {brain_number_current + 1}/{brain_number_total} to anatomical volume '
                  f'using {config.anat_align_cost_function} cost function.')

        # Align to anatomical
        if config.anat_align_cost_function == 'BBR':
            white_matter_thresholded = fsl_functions(*pack_vars,
                                                     'maths.Threshold',
                                                     self.white_matter_segmentation,
                                                     "white_matter_thresholded_")

            white_matter_binarised = fsl_functions(*pack_vars,
                                                   'maths.UnaryMaths',
                                                   white_matter_thresholded,
                                                   "", 'binarise_wmseg')

            fmri_to_anat_mat = fsl_functions(*pack_vars, 'EpiReg',
                                             fMRI_volume, "to_anat_from_",
                                             self.anat_brain, self.anat_head,
                                             white_matter_binarised)

        elif config.anat_align_cost_function == 'Correlation Ratio':
            fmri_to_anat_mat = fsl_functions(*pack_vars, 'FLIRT', fMRI_volume, "to_anat_from_", self.anat_brain)

        else:
            raise ValueError('fMRI to anatomical registration cost function type not valid.')

        # Combine fMRI-anat and anat-mni matrices
        mat = fsl_functions(*pack_vars, 'ConvertXFM', fmri_to_anat_mat, 'combined_mat_', 'concat_xfm')

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
        roi_temp_store, roi_results, idxMNI, idxBrain = self.roi_stats_setup()

        # Combine information from fMRI and MNI brains (both in native space) to assign an ROI to each voxel
        roi_temp_store, roi_results, non_segmented_volume = self.compile_voxel_values(roi_temp_store,
                                                                                      roi_results,
                                                                                      idxMNI,
                                                                                      idxBrain,
                                                                                      GM_bool)

        warnings.filterwarnings('ignore')  # Ignore warnings that indicate an ROI has only nan values

        if config.grey_matter_segment:
            volumes = {'noROI': non_segmented_volume, 'nonGreyMatter': roi_temp_store}

        else:
            volumes = {'noROI': roi_temp_store}

        roi_results, roi_temp_store = self.noise_correct_data_and_create_excluded_voxel_maps(roi_results,
                                                                                             roi_temp_store, volumes)

        # Compile ROI statistics
        roi_results = self.compile_roi_stats(roi_temp_store, roi_results)

        warnings.filterwarnings('default')  # Reactivate warnings

        # if config.bootstrap: # TODO: uncomment this when bootstrapping reimplemented
        #     roi_results = roi_stats_bootstrap(roi_temp_store, roi_results, roiNum, brain_number_current,
        #                                           brain_number_total)  # Bootstrapping

        reformat_and_save_raw_data(roi_temp_store, self._labelArray,
                                   self.save_location, self.parameters,
                                   session_number=self.session_number)  # Save results

        self.save_roi_results(roi_results)

        # Retain variables for atlas_scale function
        self.roi_results = roi_results
        self.roi_temp_store = roi_temp_store

    def save_roi_results(self, roi_results):
        headers = ['Voxels', 'Mean', 'Std_dev',
                   f'Conf_Int_{Environment_Setup.conf_level_list[int(config.conf_level_number)][0]}',
                   'Median', 'Min', 'Max', 'Excluded_Voxels']

        # Save results as dataframe
        results = pd.DataFrame(data=roi_results,
                               index=headers,
                               columns=self._labelArray)

        summary_results_path = f"{self.save_location}Summarised_results/"
        Utils.check_and_make_dir(summary_results_path)

        with open(f"{summary_results_path}{self.parameters}_ps{self.session_number}.json", 'w') as file:
            json.dump(results.to_dict(), file, indent=2)

    def compile_roi_stats(self, roi_temp_store, roi_results):
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
            roi_results[2, write_start:write_end] = np.nanstd(roi_temp_store[read_start:, :], axis=axis, ddof=1)
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

    def noise_correct_data_and_create_excluded_voxel_maps(self, roi_results, roi_temp_store, volumes):
        header, statmap_shape, save_location, stage, excluded_voxels_file_location = self.create_excluded_rois_volume(
            volumes)

        if config.verbose:
            print_outlier_removal_methods(config, self.no_ext_brain)

        # Remove outliers from ROIs
        if config.noise_cutoff:
            roi_results, roi_temp_store, self.noise_threshold = self.noise_threshold_outlier_detection(roi_results,
                                                                                                       roi_temp_store)

            excluded_voxels_file_location = f"{save_location}ExcludedVoxStage{stage}_{self.no_ext_brain}_noiseThreshOutliers.nii.gz"

            create_no_roi_volume(roi_temp_store,
                                 excluded_voxels_file_location,
                                 statmap_shape,
                                 header)

            stage += 1

        if config.gaussian_outlier_detection:
            roi_results, roi_temp_store, self.lower_gaussian_threshold, self.upper_gaussian_threshold = gaussian_outlier_detection(
                roi_results,
                roi_temp_store,
                config
            )

            excluded_voxels_file_location = f"{save_location}ExcludedVoxStage{stage}_{self.no_ext_brain}_gaussianThreshOutliers.nii.gz"

            create_no_roi_volume(roi_temp_store,
                                 excluded_voxels_file_location,
                                 statmap_shape,
                                 header)

            stage += 1

        roi_temp_store, roi_results = self.create_final_atlas_mapping(excluded_voxels_file_location,
                                                                      roi_results,
                                                                      save_location)

        return roi_results, roi_temp_store

    def create_final_atlas_mapping(self, final_excluded_voxels_volume, roi_results, output_folder):
        # Make copy of fMRI volume
        shutil.copy(self.brain, f"{output_folder}/fMRI_volume.nii.gz")

        # Make copy of mni_to_fmri brain
        orig_mni_loc = f"{output_folder}/orig_mni_to_{self.no_ext_brain}.nii.gz"
        shutil.copy(self.mni_brain, orig_mni_loc)

        # Create binary mask using -binv
        binary_mask = fsl_functions(self, output_folder, self.no_ext_brain, 'maths.UnaryMaths',
                                    final_excluded_voxels_volume, 'binary_mask_', 'binarise_and_invert')

        # Fill holes in binary mask using -fillh
        binary_mask_filled = fsl_functions(self, output_folder, self.no_ext_brain, 'maths.UnaryMaths',
                                           binary_mask, 'binary_mask_filled_', 'fill_holes')

        # Multiply mni_to_fMRI with binary_mask_filled
        fsl_functions(self, output_folder, self.no_ext_brain, 'maths.BinaryMaths',
                      self.mni_brain, 'final_mni_to_', binary_mask_filled, 'mul', 'Save output to self')

        # Calculate how many voxels have been filled in for each ROI. This will be subtracted from the excluded voxels
        # row of roi_results to create an accurate count after rerunning the analysis
        roi_results = self.correct_excluded_voxel_amount_by_number_of_filled_voxels(binary_mask,
                                                                                    binary_mask_filled,
                                                                                    orig_mni_loc,
                                                                                    roi_results,
                                                                                    output_folder)

        # Load brains and pre-initialise arrays
        roi_temp_store, _, idxMNI, idxBrain = self.roi_stats_setup()

        # The magic loop
        for counter, roi in enumerate(idxMNI):
            roi_temp_store[int(roi), counter] = idxBrain[counter]

        return roi_temp_store, roi_results

    def correct_excluded_voxel_amount_by_number_of_filled_voxels(self, binary_mask, binary_mask_filled, mni_brain,
                                                                 roi_results, output_folder):
        # binary_mask_filled - binary_mask
        filled_voxel_volume = fsl_functions(self, output_folder, self.no_ext_brain, 'maths.BinaryMaths',
                                            binary_mask_filled, 'filled_voxels_', binary_mask, 'sub')

        filled_voxel_volume = nib.load(filled_voxel_volume)
        filled_voxel_volume = filled_voxel_volume.get_fdata()
        idxBrain = filled_voxel_volume.flatten()

        # Load atlas brain (which has been converted into native space)
        mni_brain = nib.load(mni_brain)
        mni_brain = mni_brain.get_fdata()
        idxMNI = mni_brain.flatten()

        # Find the number of unique ROIs in the atlas
        roiList = list(range(0, len(self._labelArray) - 1))
        roiNum = np.size(roiList)

        # Create arrays to store the values before and after statistics
        filled_voxels_per_roi = np.full([1, roiNum], 0)

        for counter, roi in enumerate(idxMNI):
            if idxBrain[counter] == 1.0:  # Equal to 1.0 if the voxel has been filled with -fillh
                filled_voxels_per_roi[0, int(roi)] += 1

        roi_results[-1, 1:-1] = roi_results[-1, 1:-1] - filled_voxels_per_roi[:, 1:]

        return roi_results

    def create_excluded_rois_volume(self, volumes):
        excluded_vox_save_location = f"{self.save_location}/Excluded_voxels/{self.no_ext_brain}/"
        Utils.check_and_make_dir(f"{self.save_location}/Excluded_voxels/")
        Utils.check_and_make_dir(excluded_vox_save_location)

        statmap, header = Utils.load_brain(self.stat_brain)
        statmap_shape = statmap.shape

        stage = 1

        for file_name, volume in volumes.items():
            file_location = f"{excluded_vox_save_location}/ExcludedVoxStage{stage}_{self.no_ext_brain}_{file_name}.nii.gz"

            create_no_roi_volume(
                volume,
                file_location,
                statmap_shape,
                header
            )

            stage += 1

        return header, statmap_shape, excluded_vox_save_location, stage, file_location

    def noise_threshold_outlier_detection(self, roi_results, roi_temp_store):
        # Invert binary mask using -binv
        binary_mask = fsl_functions(self, self.save_location, self.no_ext_brain, 'maths.UnaryMaths',
                                    f'{self.save_location}/bet_{self.no_ext_brain}_mask.nii.gz',
                                    'inverted_bet_mask_', 'binarise_and_invert', 'Save to file list')

        # Multiply mni_to_fMRI with binary_mask_filled
        extra_cranial_voxels = fsl_functions(self, self.save_location, self.no_ext_brain, 'maths.BinaryMaths',
                                             self.stat_brain, 'extra_cranial_voxels_', binary_mask, 'mul', 'Save to file list')

        extra_cranial_vox_volume, _ = Utils.load_brain(extra_cranial_voxels)

        # Calculate noise threshold
        noise_threshold = np.nansum(extra_cranial_vox_volume) / np.count_nonzero(extra_cranial_vox_volume)

        outlier_bool_array = roi_temp_store[1:, :] < noise_threshold
        roi_results = calculate_number_of_outliers_per_roi(outlier_bool_array, roi_results)
        roi_temp_store = remove_outliers(outlier_bool_array, roi_temp_store)

        return roi_results, roi_temp_store, noise_threshold

class MatchedBrain:
    label_array = []
    save_location = None

    def __init__(self, brain_files, parameters):
        self.brains = brain_files
        self.parameters = "_".join([f"{param_name}{param_val}" for param_name, param_val
                                    in zip(config.parameter_dict2, parameters)])

        self.participant_grouped_summarised_results = {participant: [] for participant in brain_files.keys()}
        self.ungrouped_summarised_results = []
        self.ungrouped_raw_results = []

        self.session_averaged_results = []
        self.participant_averaged_results = []

        self.excluded_voxels = None
        self.conf_int = Environment_Setup.conf_level_list[int(config.conf_level_number)]

        # Instance variables to be accessed by participant averaged function
        self.session_number = None
        self.maximum_result = None
        self.minimum_result = None
        self.total_voxels = None
        self.average_voxels = None

        # Copying class attributes here is a workaround for dill, which can't access class attributes.
        self.label_array = self.label_array
        self.save_location = self.save_location

    @classmethod
    def setup_class(cls, participant_list):
        matched_brains = cls.find_shared_params(participant_list)  # Find brains which share parameter combinations

        cls.label_array = Environment_Setup.label_array
        cls.save_location = f"{Environment_Setup.save_location}Overall/"

        Utils.check_and_make_dir(cls.save_location)
        Utils.check_and_make_dir(f"{cls.save_location}/Summarised_results/")
        Utils.check_and_make_dir(f"{cls.save_location}/Summarised_results/Session_averaged_results")
        Utils.check_and_make_dir(f"{cls.save_location}/Summarised_results/Participant_averaged_results")

        matched_brain_list = set()
        for param_combination in matched_brains:
            # Initialise matched brains
            matched_brain_list.add(MatchedBrain(
                matched_brains[param_combination],
                param_combination
            ))

        cls.assign_parameters_to_brains(matched_brain_list, participant_list)

        return matched_brain_list

    @classmethod
    def find_shared_params(cls, participant_list):
        table, _ = Utils.load_paramValues_file()

        if len(table.columns) == 4:
            # Length will be equal to 4 if no parameters have been given
            ignore_column_loc = 2
            critical_column_locs = None
        else:
            ignore_column_loc, critical_column_locs, _ = Utils.find_column_locs(table)

        matched_brains = dict()
        for row in table.itertuples(index=False):
            if ignore_column_loc and str(row[ignore_column_loc]).strip().lower() in (
                    'yes', 'y', 'true'):  # If column is set to ignore then do not include it in analysis
                continue

            elif critical_column_locs:
                values = tuple(map(row.__getitem__, critical_column_locs))

            else:
                values = tuple(['results'])

            if values not in matched_brains.keys():
                matched_brains[values] = dict()

            participant, brain_file = cls.find_brain_object(row, participant_list)

            if participant in matched_brains[values]:
                matched_brains[values][participant].append(brain_file)
            else:
                matched_brains[values][participant] = [brain_file]

        return matched_brains

    @staticmethod
    def find_brain_object(row, participant_list):
        participant = next((participant for participant in participant_list if participant.participant_name == row[0]), None)

        if not participant:
            raise FileNotFoundError(f'Subject {row[0]} exists in {config.parameter_file} but no longer '
                                    f'exists in input folder "{config.input_folder_name}".\nRemove these rows from the '
                                    f'{config.parameter_file} file, or re-add them into the input folder.')

        brain = next((brain for brain in participant.brains if brain.no_ext_brain == row[1]), None)

        if not brain:
            raise FileNotFoundError(f'{row[1]} for subject {row[0]} exists in {config.parameter_file} but no longer '
                                    f'exists in input folder "{config.input_folder_name}".\nRemove these rows from the '
                                    f'{config.parameter_file} file, or re-add them into the input folder.')

        return participant.participant_name, brain.no_ext_brain

    def compile_results(self, path, config):
        if config.verbose:
            print(f'Combining results for parameter combination: {self.parameters}')

        self.save_location = f"{path}/{self.save_location}"

        # Combine raw results
        self.ungrouped_raw_results = np.concatenate(self.ungrouped_raw_results, axis=1)

        self.create_array_to_calculate_excluded_voxels()

        self.calculate_and_save_session_averaged_results()
        self.calculate_and_save_participant_averaged_results()

        # Save results
        reformat_and_save_raw_data(self.ungrouped_raw_results, self.label_array,
                                   self.save_location, self.parameters)

        return self

    def create_array_to_calculate_excluded_voxels(self):
        # TODO is this still necessary?
        self.excluded_voxels = copy.deepcopy(self.ungrouped_summarised_results)

        for result in self.excluded_voxels:
            result[0:-1, :] = 0  # Reset roi_results and only retain excluded voxels

        # Collapse list to get total excluded voxels
        self.excluded_voxels = np.sum(self.excluded_voxels, axis=0)

    def calculate_and_save_participant_averaged_results(self):
        for participant in self.participant_grouped_summarised_results:
            for session in self.participant_grouped_summarised_results[participant]:
                for row in session:
                    row[row == 0] = np.nan

        row_labels = ['Total voxels',
                      'Excluded voxels',
                      'Average voxels per session',
                      'Mean',
                      'Std_dev',
                      f'Conf_Int_{self.conf_int[0]}',
                      'Median',
                      'Minimum',
                      'Maximum',
                      'Participants',
                      'Sessions'
                      ]

        results = np.full((len(row_labels), len(self.label_array)), np.inf)

        results[0, :] = self.total_voxels
        results[1, :] = self.excluded_voxels[7]
        results[2, :] = self.average_voxels
        results[7, :] = self.minimum_result
        results[8, :] = self.maximum_result
        results[10, :] = self.session_number

        # Ignore runtime warnings about "Mean of empty slice" or "Degrees of freedom <= 0 for slice"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            participant_mean = []
            participant_median = []
            for participant in self.participant_grouped_summarised_results:
                participant_mean.append(np.nanmean(self.participant_grouped_summarised_results[participant], axis=0)[1])
                participant_median.append(
                    np.nanmedian(self.participant_grouped_summarised_results[participant], axis=0)[4])

            results[3, :] = np.nanmean(participant_mean, axis=0)
            results[4, :] = np.nanstd(participant_mean, axis=0)
            results[6, :] = np.nanmedian(participant_median, axis=0)

            # Counting nonzero as sometimes an ROI won't show up for a particular participant
            results[9, :] = np.count_nonzero(~np.isnan(participant_mean), axis=0)  # Participants
            results[5, :] = self.conf_int[1] * results[4, :] / np.sqrt(results[9, :])  # Confidence interval

        results_df = pd.DataFrame(index=pd.Index(row_labels), data=results, columns=self.label_array)

        with open(f"{self.save_location}/Summarised_results/Participant_averaged_results/{self.parameters}.json", 'w') as file:
            json.dump(results_df.to_dict(), file, indent=2)

        self.participant_averaged_results = results_df.to_numpy()

        return results_df

    def calculate_and_save_session_averaged_results(self):
        nan_data = copy.deepcopy(self.ungrouped_summarised_results)

        for session in nan_data:
            for row in session:
                row[row == 0] = np.nan

        row_labels = ['Total voxels',
                      'Excluded voxels',
                      'Average voxels per session',
                      'Mean',
                      'Std_dev',
                      f'Conf_Int_{self.conf_int[0]}',
                      'Median',
                      'Minimum',
                      'Maximum',
                      'Sessions']

        results = np.full((len(row_labels), len(self.label_array)), np.inf)

        self.total_voxels = results[0, :] = np.nansum(self.ungrouped_summarised_results, axis=0)[0]  # Total voxels
        results[1, :] = self.excluded_voxels[7]  # Excluded voxels

        # Ignore runtime warnings about "Mean of empty slice" or "Degrees of freedom <= 0 for slice"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            self.average_voxels = results[2, :] = np.nanmean(nan_data, axis=0)[0]  # Average voxels
            results[3, :] = np.nanmean(nan_data, axis=0)[1]  # Mean
            results[4, :] = np.nanstd(nan_data, axis=0)[1]  # Standard deviation
            results[6, :] = np.nanmedian(nan_data, axis=0)[4]  # Median

        self.minimum_result = results[7, :] = np.nanmin(self.ungrouped_summarised_results, axis=0)[5]  # Minimum
        self.maximum_result = results[8, :] = np.nanmax(self.ungrouped_summarised_results, axis=0)[6]  # Maximum
        self.session_number = results[9, :] = np.count_nonzero(self.ungrouped_summarised_results, axis=0)[0]  # Sessions

        results[5, :] = self.conf_int[1] * results[4, :] / np.sqrt(results[9, :])  # Confidence interval

        results_df = pd.DataFrame(index=pd.Index(row_labels), data=results, columns=self.label_array)

        with open(f"{self.save_location}/Summarised_results/Session_averaged_results/{self.parameters}.json", 'w') as file:
            json.dump(results_df.to_dict(), file, indent=2)

        self.session_averaged_results = results_df.to_numpy()

    @classmethod
    def assign_parameters_to_brains(cls, matched_brains, participant_list):
        brain_list = []
        for participant in participant_list:
            for brain in participant.brains:
                brain_list.append(brain)

        for parameter_comb in matched_brains:
            for brain in brain_list:
                try:
                    if brain.no_ext_brain in parameter_comb.brains[brain.participant_name]:
                        brain.parameters = parameter_comb.parameters

                except KeyError:
                    pass

        # Assign session number to brains if participants have more than one scan with the same parameter combination
        for parameter_comb in matched_brains:
            for subj in parameter_comb.brains:
                if len(parameter_comb.brains[subj]) > 1:
                    session_count = 1

                    for brain in brain_list:
                        try:
                            if brain.participant_name == subj and \
                                    brain.no_ext_brain in parameter_comb.brains[brain.participant_name]:
                                brain.session_number = session_count
                                session_count += 1

                        except KeyError:
                            pass

    def atlas_scale(self, max_roi_stat, brain_number_current, brain_number_total, statistic_num, atlas_path, data,
                    config):
        """Produces up to three scaled NIFTI files. Within brains, between brains (based on rois), between brains
        (based on the highest seen value of all brains and rois)."""

        if data == 'Session averaged':
            results = self.session_averaged_results
            subfolder = 'Session_averaged_results/'
            statistic_labels = config.statistic_options['Session averaged']

        else:
            results = self.participant_averaged_results
            subfolder = 'Participant_averaged_results/'
            statistic_labels = config.statistic_options['Participant averaged']

        if config.verbose and max(max_roi_stat) != 0.0:
            print(f'Creating {statistic_labels[statistic_num]} NIFTI_ROI file for parameter combination '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.parameters}.')

        elif config.verbose \
                and statistic_labels[statistic_num] == 'Excluded_voxels_amount' \
                and max(max_roi_stat) == 0.0:
            print(f'Not creating {statistic_labels[statistic_num]} NIFTI_ROI file for parameter combination '
                  f'{brain_number_current + 1}/{brain_number_total}: {self.parameters} as no voxels have been excluded.')

            return

        self.scale_and_save_atlas_images(atlas_path, max_roi_stat, results, statistic_num,
                                         f"{self.save_location}NIFTI_ROI/{subfolder}",
                                         f"{self.parameters}_{statistic_labels[statistic_num]}")

    @classmethod
    def scale_and_save_atlas_images(cls, atlas_path, max_stat, results, statistic_num, file_path, file_name):
        if type(max_stat) is np.ndarray:
            roi_scaled_stat = [(y / x) * 100 for x, y in zip(max_stat, results[statistic_num, :])]

            # Find maximum statistic value (excluding No ROI and overall category)
            global_scaled_stat = [(y / max(max_stat[1:-1])) * 100 for y in results[statistic_num, :]]
        else:
            roi_scaled_stat, global_scaled_stat = None, None

        atlas = nib.load(atlas_path)
        affine = atlas.affine
        atlas = atlas.get_fdata()

        unscaled_stat, within_roi_stat, mixed_roi_stat = cls.group_roi_stats(atlas, global_scaled_stat,
                                                                             roi_scaled_stat, statistic_num,
                                                                             results)
        # Convert atlas to NIFTI and save it
        scale_stats = [
            (unscaled_stat, f"{file_path}/{file_name}.nii.gz"),
            (within_roi_stat, f"{file_path}/{file_name}_within_roi_scaled.nii.gz"),
            # TODO reimplement mixed_roi_stat
            # (mixed_roi_stat, f"{file_path}/{file_name}_mixed_roi_scaled.nii.gz")
        ]

        for scale_stat in scale_stats:
            if scale_stat[0] is not None:
                type(scale_stat[0]) # todo: was getting error from np ndarray being dtype object, change type conversion to whatever is output by this
                scaled_brain = nib.Nifti1Image(np.float64(scale_stat[0]), affine)
                scaled_brain.to_filename(scale_stat[1])

    @staticmethod
    def group_roi_stats(atlas, global_scaled_stat, roi_scaled_stat, statistic_num, results):
        # Iterate through each voxel in the atlas
        atlas = atlas.astype(int)

        # Assign stat values for each ROI all at once
        unscaled_stat = results[statistic_num, atlas]
        # Make ROI group 0 (No ROI) nan so it does not effect colourmap when viewing in fsleyes
        unscaled_stat[atlas == 0] = np.nan

        if global_scaled_stat and roi_scaled_stat:
            within_roi_stat = np.array(roi_scaled_stat)[atlas]
            mixed_roi_stat = np.array(global_scaled_stat)[atlas]

            # Make ROI group 0 (No ROI) nan so it does not effect colourmap when viewing in fsleyes
            within_roi_stat[atlas == 0] = np.nan
            mixed_roi_stat[atlas == 0] = np.nan
        else:
            within_roi_stat = None
            mixed_roi_stat = None

        return unscaled_stat, within_roi_stat, mixed_roi_stat


def create_no_roi_volume(volume, save_location, statmap_shape, header):
    data = volume[0, :].copy()
    data = data.reshape(statmap_shape)

    brain = nib.Nifti1Pair(data, None, header)
    nib.save(brain, save_location)


def print_outlier_removal_methods(config, brain):
    string = []
    if config.noise_cutoff:
        string.append('noise cutoff')
    if config.gaussian_outlier_detection:
        string.append('gaussian outlier detection')

    if string:
        print(f'Running outlier removal using {" & ".join(string)} for volume: {brain}')


def run_flirt_cost_function(fslfunc, ref, init, out_file, matrix_file, config, wmseg=None):
    fslfunc.inputs.reference = ref
    fslfunc.inputs.args = f"-init {init}"  # args used as in_matrix_file method not working
    fslfunc.inputs.out_file = out_file
    fslfunc.inputs.out_matrix_file = matrix_file

    if wmseg:
        fslfunc.inputs.cost = 'bbr'
        fslfunc.inputs.wm_seg = wmseg

    if config.verbose_cmd_line_args:
        print(f"{fslfunc.cmdline}\n")

    output = fslfunc.run()
    cost_func = float(re.search("[0-9]*\.[0-9]+", output.runtime.stdout)[0])

    # Clean up files
    os.remove(out_file)
    os.remove(matrix_file)

    return cost_func


def fsl_functions(obj, save_location, no_ext_brain, func, input_volume, prefix, *argv):
    """Run an FSL function using NiPype."""
    save_override = False
    current_mat = None
    current_brain, fslfunc, suffix = fsl_functions_setup(func, input_volume, no_ext_brain, prefix, save_location)

    # Arguments dependent on FSL function used
    if func == 'MCFLIRT':
        fslfunc.inputs.save_rms = True

    elif func == 'BET':
        fslfunc.inputs.functional = True

    elif func == 'FLIRT':
        fslfunc.inputs.reference = argv[0]
        fslfunc.inputs.dof = config.dof
        current_mat = fslfunc.inputs.out_matrix_file = f'{save_location}{prefix}{no_ext_brain}.mat'

    elif func == 'maths.Threshold':
        fslfunc.inputs.thresh = 0.5

    elif func == 'maths.UnaryMaths':
        if argv[0] == 'binarise_wmseg':
            fslfunc.inputs.operation = 'bin'
            current_brain = fslfunc.inputs.out_file = f"{save_location}{prefix}{no_ext_brain}_fast_wmseg.nii.gz"

        elif argv[0] == 'binarise_and_invert':
            fslfunc.inputs.operation = 'binv'

            try:
                if argv[1] == 'Save to file list':
                    save_override = True
            except IndexError:
                pass

        elif argv[0] == 'fill_holes':
            fslfunc.inputs.operation = 'fillh'

    elif func == 'maths.BinaryMaths':
        fslfunc.inputs.operand_file = argv[0]
        fslfunc.inputs.operation = argv[1]

        try:
            if argv[2] == 'Save output to self':
                obj.mni_brain = current_brain
            elif argv[2] == 'Save to file list':
                save_override = True
        except IndexError:
            pass

    elif func == 'EpiReg':
        fslfunc.inputs.epi = input_volume
        fslfunc.inputs.t1_brain = argv[0]
        fslfunc.inputs.t1_head = argv[1]
        fslfunc.inputs.wmseg = argv[2]

        # Gets around a bug in nipype code that only allows the wmseg value to be the outbase value with _fast_wmseg suffix
        fslfunc.inputs.out_base = argv[2].replace('_fast_wmseg.nii.gz', '')
        current_mat = f'{fslfunc.inputs.out_base}.mat'
        current_brain = f'{fslfunc.inputs.out_base}.nii.gz'

    elif func == 'FAST':
        fslfunc.inputs.in_files = input_volume
        fslfunc.inputs.out_basename = save_location

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
        print(f"{fslfunc.cmdline}\n")

    fslfunc.run()

    if func not in ('maths.Threshold', 'maths.UnaryMaths', 'maths.BinaryMaths', 'FAST') or save_override:
        fsl_function_file_handle(current_brain, current_mat, func, no_ext_brain, obj, prefix, save_location, suffix)

    if func == 'FLIRT':
        return current_mat
    elif func == 'EpiReg':
        return f'{save_location}{prefix}{no_ext_brain}.mat'
    elif func == 'FAST':
        return

    return current_brain


def fsl_function_file_handle(current_brain, current_mat, func, no_ext_brain, obj, prefix, save_location, suffix):
    if func in ('FLIRT', 'ApplyXFM'):
        obj.file_list.extend([current_brain, current_mat])

    elif func == 'EpiReg':
        new_base = f'{save_location}{prefix}{no_ext_brain}'
        renamed_brain = f'{new_base}.nii.gz'
        renamed_mat = f'{new_base}.mat'
        renamed_wmedge = f'{new_base}_fast_wmedge.nii.gz'
        renamed_wmseg = f'{new_base}_fast_wmseg.nii.gz'

        os.rename(current_brain, renamed_brain)
        os.rename(current_mat, renamed_mat)
        os.rename(f'{save_location}{no_ext_brain}_fast_wmedge.nii.gz', renamed_wmedge)
        os.rename(f'{save_location}{no_ext_brain}_fast_wmseg.nii.gz', renamed_wmseg)

        obj.file_list.extend([renamed_brain, renamed_mat, renamed_wmedge, renamed_wmseg])

    elif func == 'BET':
        obj.file_list.extend([current_brain, f"{save_location}{prefix}{no_ext_brain}_mask{suffix}"])

    elif func == 'MCFLIRT':
        # Find all the motion correction files that are not the actual brain volume
        mc_files = [direc for direc in os.listdir(f"{save_location}motion_correction_files/")
                    if re.search(f'^{prefix}{no_ext_brain}{suffix}(?!$)', direc)]

        # Remove .nii.gz from middle of string
        for file in mc_files:
            os.rename(f"{save_location}motion_correction_files/{file}",
                      f"{save_location}motion_correction_files/{file.replace(suffix, '')}")
    else:
        obj.file_list.append(current_brain)


def fsl_functions_setup(func, input, no_ext_brain, prefix, save_location):
    if 'maths' in func:
        func = func.split('.')[1]
        fslfunc = getattr(maths, func)()
    else:
        fslfunc = getattr(fsl, func)()

    # Standard variables that may be changed for specific FSL functions
    suffix = '.nii.gz'
    fslfunc.inputs.output_type = 'NIFTI_GZ'

    current_brain = None
    if func not in ('EpiReg', 'FAST'):
        fslfunc.inputs.in_file = input
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


def reformat_and_save_raw_data(roi_temp_store, labelArray, save_location, no_ext_brain, session_number=None):
    if session_number is None:
        session_suffix = ''
    else:
        session_suffix = f'_ps{session_number}'

    formatted_roi_temp_store = roi_temp_store.copy()  # Copy roi_temp_store so changes do get saved later

    # Reorganise matrix to later remove nan rows
    formatted_roi_temp_store = formatted_roi_temp_store.transpose()
    i = np.arange(formatted_roi_temp_store.shape[1])
    # Find indices of nans and put them at the end of each column
    a = np.isnan(formatted_roi_temp_store).argsort(0, kind='mergesort')
    # Reorganise matrix with nans at end
    formatted_roi_temp_store[:] = formatted_roi_temp_store[a, i]

    raw_results = pd.DataFrame(data=formatted_roi_temp_store,
                               columns=labelArray[:-1])

    # Remove the required rows from the dataframe
    raw_results = raw_results.drop(raw_results.columns[0], axis=1)

    # Remove rows where all columns have NaNs (essential to keep file size down)
    raw_results = raw_results.dropna(axis=0, how='all')

    # Convert to dict and get rid of row numbers to significantly decrease file size
    roidict = Utils.dataframe_to_dict(raw_results)

    raw_results_path = f"{save_location}Raw_results/"
    Utils.check_and_make_dir(raw_results_path)

    with open(f"{raw_results_path}{no_ext_brain}{session_suffix}_raw.json", 'w') as file:
        json.dump(roidict, file, indent=2)


def verify_param_values():
    """Compare critical parameter choices to those in paramValues.csv. Exit with exception if discrepancy found."""
    table, _ = Utils.load_paramValues_file()
    Utils.find_column_locs(table)


def construct_combined_results(directory, analysis_type='overall', subfolder=''):
    if subfolder == 'session averaged':
        directory = f"{directory}/Summarised_results/Session_averaged_results"
    elif subfolder == 'participant averaged':
        directory = f"{directory}/Summarised_results/Participant_averaged_results/"
    else:
        directory = f"{directory}/Summarised_results/"

    json_file_list = [os.path.basename(f) for f in glob(f"{directory}/*.json")]

    save_combined_results_file(directory, json_file_list)

    if analysis_type == 'participant':
        save_averaged_results_file(directory, json_file_list)


def save_averaged_results_file(directory, json_file_list):
    Utils.check_and_make_dir(f"{directory}/Averaged_results/")

    parameter_dict = {}
    for jsn in json_file_list:
        if 'combined_results' in jsn:
            continue

        param, _, _ = jsn.partition('_ps')

        if param not in parameter_dict:
            parameter_dict[param] = [jsn]
        else:
            parameter_dict[param].append(jsn)

    overall_averaged_dataframe = pd.DataFrame()
    for parameter, jsns in parameter_dict.items():
        session_averaged_dataframe = pd.DataFrame()

        for jsn in jsns:
            with open(f'{directory}/{jsn}', "r") as results:
                results = pd.DataFrame(json.load(results))
                row_order = results.index

            session_averaged_dataframe = pd.concat((session_averaged_dataframe, results))

        # Get the summarised results across all json files for each parameter combination
        grouped_by_stat = session_averaged_dataframe.groupby(session_averaged_dataframe.index)
        averaged_dataframe = grouped_by_stat.mean().reindex(row_order)

        overall_averaged_dataframe = pd.concat((overall_averaged_dataframe, averaged_dataframe))

        with open(f"{directory}/Averaged_results/{parameter}.json", 'w') as file:
            json.dump(averaged_dataframe.to_dict(), file, indent=2)

    # Repeat the above to create the overall summarised results
    grouped_by_stat = overall_averaged_dataframe.groupby(overall_averaged_dataframe.index)
    averaged_dataframe = grouped_by_stat.mean().reindex(row_order)

    with open(f"{directory}/Averaged_results/overall.json", 'w') as file:
        json.dump(averaged_dataframe.to_dict(), file, indent=2)


def save_combined_results_file(directory, json_file_list):
    session_dict = split_jsons_by_session(json_file_list)

    for session, jsns in session_dict.items():
        if not jsns:
            continue

        combined_dataframe = pd.DataFrame()

        for jsn in jsns:
            if 'combined_results' in jsn:
                continue

            # Splits a file name. For example from hb1_ip2.json into ['hb1', 'ip2']
            parameters = list(filter(None, re.split(f'_|ps[0-9]*|.json', jsn)))

            # Remove critical parameter name from file name. For example turn ['hb1', 'ip2'] into [1, 2]
            for counter, critical_parameter in enumerate(config.parameter_dict2):
                parameters[counter] = parameters[counter].replace(critical_parameter, '')

            current_dataframe = pd.read_json(f"{directory}/{jsn}")
            current_dataframe = current_dataframe.transpose()

            for counter, parameter_name in enumerate(config.parameter_dict):
                current_dataframe[parameter_name] = parameters[counter]  # Add parameter columns

            current_dataframe['File_name'] = Utils.strip_ext(jsn)

            if combined_dataframe.empty:
                combined_dataframe = current_dataframe

            else:
                combined_dataframe = pd.concat([combined_dataframe, current_dataframe], sort=True)

        if session == 0:
            session_suffix = ''
        else:
            session_suffix = f'_ps{session}'

        # Save combined results
        combined_dataframe = combined_dataframe.reset_index()
        combined_dataframe.to_json(f"{directory}/combined_results{session_suffix}.json", orient='records', indent=2)


def split_jsons_by_session(json_file_list):
    session_dict = {0: []}

    for jsn in json_file_list:
        if '_ps' in jsn:
            session_number = re.findall('_ps[0-9]*', jsn)[0].split('_ps')[-1]

            if session_number not in session_dict:
                session_dict[session_number] = []

            session_dict[session_number].append(jsn)

        else:
            session_dict[0].append(jsn)

    return session_dict


def gaussian_outlier_detection(roi_results, roi_temp_store, config):
    # Convert gaussian outlier contamination into confidence interval to retain for scipy norm.interval function
    confidence_interval = 1 - config.gaussian_outlier_contamination

    # Fit a gaussian to the data using EllipticEnvelope
    outliers, lower_gaussian_threshold, upper_gaussian_threshold = outlier_detection_using_gaussian(
        data=roi_temp_store[1:, :],
        contamination=confidence_interval,
        config=config
    )

    outliers.sort()  # Sort to make it easier to find start and end

    outlier_bool_array = np.isin(roi_temp_store[1:, :], outliers)
    roi_results = calculate_number_of_outliers_per_roi(outlier_bool_array, roi_results)
    roi_temp_store = remove_outliers(outlier_bool_array, roi_temp_store)

    return roi_results, roi_temp_store, lower_gaussian_threshold, upper_gaussian_threshold


def outlier_detection_using_gaussian(data, contamination, config):
    values = data.copy()
    values = values[~np.isnan(values)]

    prediction = norm(*norm.fit(values))
    gaussian_lims = prediction.interval(contamination)

    lower_gaussian_threshold = None
    upper_gaussian_threshold = None

    if config.gaussian_outlier_location == 'below gaussian':
        outliers = [value for value in values if value < gaussian_lims[0]]
        lower_gaussian_threshold = gaussian_lims[0]

    elif config.gaussian_outlier_location == 'above gaussian':
        outliers = [value for value in values if value > gaussian_lims[1]]
        upper_gaussian_threshold = gaussian_lims[1]

    elif config.gaussian_outlier_location == 'both':
        outliers = [value for value in values if value < gaussian_lims[0] or value > gaussian_lims[1]]
        lower_gaussian_threshold = gaussian_lims[0]
        upper_gaussian_threshold = gaussian_lims[1]

    else:
        raise Exception('gaussian_outlier_location not valid')

    return outliers, lower_gaussian_threshold, upper_gaussian_threshold


def calculate_number_of_outliers_per_roi(outlier_bool_array, roi_results):
    # Count how many voxels are below noise threshold for each ROI
    below_threshold = np.count_nonzero(outlier_bool_array, axis=1)
    # Assign voxels below threshold to excluded voxels column
    roi_results[7, 1:-1] += below_threshold

    return roi_results


def remove_outliers(outlier_bool_array, roi_temp_store):
    # Convert from bool to float64 to be able to assign values
    outlier_float_array = np.max(outlier_bool_array, axis=0).astype('float64')

    # Replace False with nans and True with statistic value
    outlier_float_array[outlier_float_array == 0.0] = np.nan

    # Use the transpose of roi_temp_store and outlier bool array to make sure output is ordered by columns not rows
    outlier_float_array[outlier_float_array == 1.0] = roi_temp_store[1:, :].T[outlier_bool_array.T]

    # Set any outlier ROI values to nan
    roi_temp_store[1:, :][outlier_bool_array] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        # Stack no ROI values on top of array containing outlier values, and then use nan max to condense into one array
        roi_temp_store[0, :] = np.nanmax(np.vstack((outlier_float_array, roi_temp_store[0, :])), axis=0)

    return roi_temp_store


def json_search():
    if len(json_file_list) == 0:
        raise NameError('Folder selection error. Could not find json files in the "Summarised_results" directory.')
    else:
        return json_file_list

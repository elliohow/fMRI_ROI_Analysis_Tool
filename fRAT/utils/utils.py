import argparse
import ast
import os
import re
import shutil
import sys
from glob import glob
from pathlib import Path
from tkinter import Tk, filedialog
from types import SimpleNamespace

import multiprocess as mp
import nibabel as nib
import pandas as pd
import toml

from utils.fRAT_config_setup import *

config = None


class Utils:
    @staticmethod
    def convert_toml_input_to_python_object(x):
        try:
            if x in ['true', 'false']:
                x = x.title()

            x = ast.literal_eval(x)

            if isinstance(x, tuple):
                x = list(x)

        except (ValueError, SyntaxError):
            pass

        return x

    @staticmethod
    def argparser():
        # Create the parser
        parser = argparse.ArgumentParser(prog='fRAT.py', usage='%(prog)s [arguments]',
                                         description='Convert voxelwise statistics to regionwise statistics for fMRI '
                                                     'data.',
                                         epilog='Supplying arguments in this way is for advanced users only. It is '
                                                'recommended that settings are changed using the GUI or by '
                                                'editing the fRAT_config.toml file. '
                                                'Arguments should be given in toml format. For example: '
                                                'true should be used instead of True and strings should be in '
                                                'quotation marks. Where a comma-separated list is supposed to be '
                                                'given, this should be in the format: ["mb", "sense"].')

        # Add the arguments
        parser.add_argument('--make_table', dest='make_table',
                            help='true or false. Use this flag to create a csv file to store parameter information '
                                 'about files. Recommended that this file is created and filled in before fRAT '
                                 'execution.')

        arg_categories = pages[1:]

        for category in arg_categories:
            for arg in eval(category):
                try:
                    help_text = eval(category)[arg]["Description"].replace("%", "%%")
                except KeyError:
                    continue

                if help_text == "":
                    help_text = f"Recommended value: {eval(category)[arg]['Recommended']}"

                parser.add_argument(f'--{arg}', dest=arg, help=help_text)

        # Execute the parse_args() method
        args = parser.parse_args()

        return args

    @staticmethod
    def find_files(directory, *extensions):
        files = []
        for extension in extensions:
            if extension[0] == ".":
                extension.lstrip(".")

            these_files = [os.path.basename(f) for f in glob(f"{directory}/*.{extension}")]

            if these_files:
                files.extend(these_files)

        return files

    @staticmethod
    def dataframe_to_dict(dataframe):
        roidict = {}
        for column in dataframe.columns:
            roidict[column] = dataframe[column].dropna().to_numpy().tolist()

        return roidict

    @staticmethod
    def dict_to_dataframe(roidict):
        return pd.DataFrame.from_dict(roidict, orient='index').transpose()

    @staticmethod
    def file_browser(title='', chdir=False):
        root = Tk()
        root.withdraw()  # Hide tkinter root window

        directory = filedialog.askdirectory(title=title)

        root.destroy()  # Destroy tkinter root window

        if not directory:
            raise FileNotFoundError('No folder selected.')

        if chdir:
            os.chdir(directory)

        if config is None or config.verbose:
            print(f"Selected directory: {directory}")

        return directory

    @staticmethod
    def print_and_save(file_object, print_to_terminal, *args):
        for line in args:
            if print_to_terminal:
                print(line)
            file_object.write(str(line) + "\n")

    @staticmethod
    def save_config(newdir, config_file, additional_info=None, relevant_sections='all', config_name='config_log'):
        with open(f'{newdir}/{config_name}.toml', 'w') as f, open(f'{config_file}.toml', 'r') as r:
            if additional_info:
                for line in additional_info:
                    f.write(line)

                f.write('\n\n')

            current_section = None
            for line in r:
                if line[0] == '#':
                    current_section = line.replace('#', '')[1:-1]

                if relevant_sections == 'all':
                    f.write(line)
                elif current_section == 'Version Info' or current_section in relevant_sections:
                    f.write(line)

    @staticmethod
    def move_file(old_name, original_dir, new_dir, copy=False, rename_copy=True, parameter_file=False):
        if not original_dir.endswith('/'):
            original_dir += '/'

        if not new_dir.endswith('/'):
            new_dir += '/'

        if parameter_file:
            new_name = "paramValues.csv"
        else:
            new_name = old_name

        if copy:
            if rename_copy:
                shutil.copy(f"{original_dir}{old_name}", f"{new_dir}copy_{new_name}")
            else:
                shutil.copy(f"{original_dir}{old_name}", f"{new_dir}{new_name}")
        else:
            os.rename(f"{original_dir}{old_name}", f"{new_dir}{new_name}")

    @staticmethod
    def check_and_make_dir(path, delete_old=False):
        if delete_old and os.path.exists(path):
            shutil.rmtree(path)
            Utils.mk_dir(path)

        elif not os.path.exists(path):
            Utils.mk_dir(path)

    @staticmethod
    def read_combined_results(folder, averaging_type):
        df = pd.DataFrame()
        path = ''

        try:
            if averaging_type == 'Session averaged':
                path = f"{folder}/Overall/Summarised_results/Session_averaged_results/combined_results.json"
                df = pd.read_json(path)

            elif averaging_type == 'Participant averaged':
                path = f"{folder}/Overall/Summarised_results/Participant_averaged_results/combined_results.json"
                df = pd.read_json(path)

        except ValueError:
            raise Exception(f"combined_results.json not found in {config.averaging_type} folder.")

        return df, path

    @staticmethod
    def find_column_locs(table):
        table.columns = [x.lower() for x in table.columns]  # Convert to lower case for comparison to key later

        ignore_column_loc = next((counter for counter, column in enumerate(table.columns) if "ignore file" in column),
                                 False)

        baseline_column_loc = next((counter for counter, column in enumerate(table.columns)
                                    if "baseline" in column), False)
        if not baseline_column_loc:
            raise NameError(f'No baseline column found in {config.parameter_file}.')

        critical_column_locs = set()
        for key in config.parameter_dict:
            column_loc = next((counter for counter, column in enumerate(table.columns) if key.lower() == column), False)

            if column_loc:
                critical_column_locs.add(column_loc)
            else:
                raise Exception(f'Key "{key}" not found in {config.parameter_file}. Check the Critical Parameters '
                                f'option in the Parsing menu (parameter_dict1 if not using the GUI) correctly match '
                                f'the {config.parameter_file} headers.')

        return ignore_column_loc, critical_column_locs, baseline_column_loc

    @staticmethod
    def load_paramValues_file():
        if os.path.isfile(f"{os.getcwd()}/{config.parameter_file}"):
            table = pd.read_csv(config.parameter_file)  # Load param table
        else:
            try:
                table = pd.read_csv(f"copy_paramValues.csv")  # Load param table
            except FileNotFoundError:
                raise Exception('Make sure a copy of paramValues.csv is in the chosen folder. \n'
                                'Also make sure the selected folder contains all the participant directories '
                                'in the necessary BIDS format e.g. sub-01.')

        return table

    @staticmethod
    def mk_dir(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @staticmethod
    def instance_method_handler(*argv):
        return getattr(argv[0], argv[1])(*argv[2:])

    @staticmethod
    def class_method_handler(*argv):
        return argv[0](*argv[1:])

    @staticmethod
    def start_processing_pool(restart=False):
        if config.max_core_usage == 'max':
            workers = mp.cpu_count()
        else:
            workers = config.max_core_usage

        ctx = mp.get_context('forkserver')  # This stops segmentation fault for MacOS

        if config.verbose and not restart:
            print(f"\nStarting processing pool using {workers} cores.")

        return ctx.Pool(processes=workers)

    @staticmethod
    def join_processing_pool(pool, restart):
        pool.close()
        pool.join()

        if restart:
            pool = Utils.start_processing_pool(restart=True)

        return pool

    @classmethod
    def load_config(cls, config_path, filename, save=True):
        with open(f'{config_path}/{filename}', 'r') as tomlfile:
            try:
                parse = tomlfile.readlines()
                parse = toml.loads(''.join(parse))

                for key in parse:
                    if parse[key] == 'None':
                        parse[key] = None

                if save:
                    global config

                config = SimpleNamespace(**parse)

                if filename == 'fRAT_config.toml':
                    config.statistic_options = {
                        'Participant averaged': ['Total voxels',
                                                 'Excluded voxels',
                                                 'Average voxels per session',
                                                 'Mean',
                                                 'Std_dev',
                                                 'Confidence_interval',
                                                 'Median',
                                                 'Minimum',
                                                 'Maximum',
                                                 'Participants',
                                                 'Sessions'],

                        'Session averaged': ['Voxel_amount',
                                             'Excluded_voxels_amount',
                                             'Average_voxels',
                                             'Mean',
                                             'Standard_deviation',
                                             'Confidence_interval',
                                             'Median',
                                             'Minimum',
                                             'Maximum',
                                             'Sessions']
                    }

                    config.parameter_dict = {config.parameter_dict1[i]:
                                                 config.parameter_dict2[i] for i in range(len(config.parameter_dict1))}

                    Utils.clean_config_options(config)

                return config

            except (toml.decoder.TomlDecodeError, AttributeError):
                raise Exception('Config file not in correct format or missing entries.')

    @staticmethod
    def clean_config_options(config):
        atlas_options = ['Cerebellum-MNIflirt', 'Cerebellum-MNIfnirt', 'HarvardOxford-cort',
                         'HarvardOxford-sub', 'JHU-ICBM-labels', 'JHU-ICBM-tracts', 'juelich', 'MNI',
                         'SMATT-labels', 'STN',
                         'striatum-structural', 'Talairach-labels', 'Thalamus']

        if not type(config.atlas_number) is int:
            config.atlas_number = atlas_options.index(config.atlas_number)

        conf_level_options = ['80%, 1.28', '85%, 1.44', '90%, 1.64', '95%, 1.96', '98%, 2.33', '99%, 2.58']

        if not type(config.conf_level_number) is int:
            config.bootstrap_alpha = 1 - float(f"0.{re.split('%', config.conf_level_number)[0]}")
            config.conf_level_number = conf_level_options.index(config.conf_level_number)

        return config

    @staticmethod
    def load_brain(file_path):
        data = nib.load(file_path)

        header = data.header
        data = data.get_fdata()

        return data, header

    @staticmethod
    def strip_ext(path):
        extensions = ['.nii.gz', '.nii', '.hdr', '.json']

        for extension in extensions:
            path = path.replace(extension, "")

        return path

    @staticmethod
    def find_participant_dirs(directory):
        # Searches for folders that start with e.g. sub-01
        participant_path = [direc for direc in glob(f"{directory}/*") if re.search("sub-[0-9]+$", direc)]
        participant_names = [participant.split('/')[-1] for participant in participant_path]

        if len(participant_path) == 0:
            raise FileNotFoundError('Participant directories not found.\n'
                                    'Make sure participant directories are labelled e.g. sub-01 and the selected '
                                    'directory contains all participant directories.')
        elif config.verbose:
            print(f'Found {len(participant_path)} participant folders.')

        return participant_path, participant_names

    @staticmethod
    def checkversion():
        # Check Python version:
        expect_major = 3
        expect_minor = 8
        expect_rev = 0

        print(f"\nfRAT is developed and tested with Python {str(expect_major)}.{str(expect_minor)}.{str(expect_rev)}")
        if sys.version_info[:3] < (expect_major, expect_minor, expect_rev):
            current_version = f"{str(sys.version_info[0])}.{str(sys.version_info[1])}.{str(sys.version_info[2])}"
            print(f"INFO: Python version {current_version} is untested. Consider upgrading to version "
                  f"{str(expect_major)}.{str(expect_minor)}.{str(expect_rev)} if there are errors running the fRAT.")

    @staticmethod
    def chdir_to_output_directory(current_step, config):
        if current_step == 'Plotting' and config.run_statistics:
            return  # Will already be in the correct directory

        elif config.run_analysis:
            from utils.analysis import Environment_Setup
            json_directory = f'{os.getcwd()}/{Environment_Setup.save_location}'

            os.chdir(json_directory)

        elif config.report_output_folder in ("", " "):
            print('Select the directory output by the fRAT.')
            json_directory = Utils.file_browser(title='Select the directory output by the fRAT', chdir=True)

        else:
            json_directory = config.report_output_folder

            try:
                os.chdir(json_directory)
            except FileNotFoundError:
                raise FileNotFoundError(
                    'Output folder location (fRAT output folder location) in fRAT_config.toml is not a valid directory.')

            if config.verbose:
                print(f'Output folder selection: {json_directory}')

        return json_directory

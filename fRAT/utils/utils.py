import argparse
import ast
import os
import shutil
import toml
import re
from glob import glob
from tkinter import Tk, filedialog
import multiprocess as mp
import pandas as pd
import warnings
from pathlib import Path
from types import SimpleNamespace

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
                help_text = eval(category)[arg]["Description"].replace("%", "%%")

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
    def save_config(newdir, config_file):
        with open(f'{newdir}/config_log.toml', 'w') as f, open(f'{config_file}.toml', 'r') as r:
            for line in r:
                f.write(line)

    @staticmethod
    def move_file(name, original_dir, new_dir, copy=False, rename_copy=True):
        if not original_dir.endswith('/'):
            original_dir += '/'

        if not new_dir.endswith('/'):
            new_dir += '/'

        if copy:
            if rename_copy:
                shutil.copy(f"{original_dir}{name}", f"{new_dir}copy_{name}")
            else:
                shutil.copy(f"{original_dir}{name}", f"{new_dir}{name}")
        else:
            os.rename(f"{original_dir}{name}", f"{new_dir}{name}")

    @staticmethod
    def check_and_make_dir(path, delete_old=False):
        if delete_old and os.path.exists(path):
            shutil.rmtree(path)
            Utils.mk_dir(path)

        elif not os.path.exists(path):
            Utils.mk_dir(path)

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
                    config.statistic_options = ['Voxel_amount', 'Mean', 'Standard_deviation', 'Confidence_interval',
                                                'Median', 'Minimum', 'Maximum', 'Excluded_voxels_amount']

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
    def strip_ext(path):
        path = Path(path)
        extensions = "".join(path.suffixes)

        return str(path).replace(extensions, "")

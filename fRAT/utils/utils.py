import argparse
import os
import shutil
import toml
import re
import numpy as np
from glob import glob
from tkinter import Tk, filedialog
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import multiprocess as mp
import warnings
from types import SimpleNamespace

config = None


class Utils:
    @staticmethod
    def argparser():
        # Create the parser
        parser = argparse.ArgumentParser(prog='roi_analysis_tool',
                                         description='Convert voxelwise statistics to regionwise statistics for MRI data.')

        # Add the arguments
        parser.add_argument('--brain_loc', dest='brain_loc', action='store', type=str,
                            help='Directory location of brain files for analysis. '
                                 'Can be set in config_test.py to use a GUI to find folder instead).')

        parser.add_argument('--json_loc', dest='json_loc', action='store', type=str,
                            help='Directory location of json files produced by the roi_analysis_tool '
                                 '(can be set in config_test.py to use a GUI to find folder instead).')

        parser.add_argument('--make_table', dest='make_table', action='store_true',
                            help='Use this flag to create a csv file to store parameter information about files.'
                                 'Recommended that this file is created and filled in before tool execution'
                                 ' (this setting can alternatively be set to True or False in config_test.py).')

        parser.add_argument('--print_info', dest='print_info', action='store_true',
                            help='Use this flag to print a list of possible atlases and other information.')

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
    def file_browser(title='', chdir=False):
        directory = filedialog.askdirectory(title=title)

        if not directory:
            raise FileNotFoundError('No folder selected.')

        if chdir:
            os.chdir(directory)

        if config is None or config.verbose:
            print(f"Selected directory: {directory}")

        return directory

    @staticmethod
    def save_config(directory):
        with open(directory + '/config_log.toml', 'w') as f, open('config.toml', 'r') as r:
            for line in r:
                f.write(line)

    @staticmethod
    def calculate_confidence_interval(data, alpha, roi=None):
        warnings.filterwarnings('ignore', category=PendingDeprecationWarning)  # Silences a deprecation warning from bootstrapping library using outdated numpy matrix instead of numpy array

        if roi is None:
            data = data.flatten()
            values = np.array([x for x in data if str(x) != 'nan'])
        else:
            values = np.array([x for x in data[roi, :] if str(x) != 'nan'])

        results = bs.bootstrap(values, stat_func=bs_stats.mean, alpha=alpha, iteration_batch_size=10, num_threads=-1)
        conf_int = (results.upper_bound - results.lower_bound) / 2

        return results.value, conf_int

    @staticmethod
    def move_file(name, original_dir, new_dir, copy=False):
        if not original_dir.endswith('/'):
            original_dir += '/'

        if not new_dir.endswith('/'):
            new_dir += '/'

        if copy:
            shutil.copy(f"{original_dir}{name}", f"{new_dir}copy_{name}")
        else:
            os.rename(f"{original_dir}{name}", f"{new_dir}{name}")

    @staticmethod
    def check_and_make_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def instance_method_handler(*argv):
        return getattr(argv[0], argv[1])(*argv[2:])

    @staticmethod
    def class_method_handler(*argv):
        return argv[0](*argv[1:])
        # Histogram calculations take quite some time, quicker way to do this too?

    @staticmethod
    def start_processing_pool():
        if config.multicore_processing:
            if config.max_core_usage == 'max':
                workers = mp.cpu_count()
            else:
                workers = config.max_core_usage

            ctx = mp.get_context('forkserver')  # This stops segmentation fault for MacOS
            if config.verbose:
                print(f"\nStarting processing pool using {workers} cores.")
            return ctx.Pool(processes=workers)

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

                # Cleans config output
                atlas_options = ['Cerebellum-MNIflirt', 'Cerebellum-MNIfnirt', 'HarvardOxford-cort',
                                 'HarvardOxford-sub', 'JHU-ICBM-labels', 'JHU-ICBM-tracts', 'juelich', 'MNI',
                                 'SMATT-labels', 'STN',
                                 'striatum-structural', 'Talairach-labels', 'Thalamus']
                config.atlas_number = atlas_options.index(config.atlas_number)

                roi_stat_options = ['Voxel number', 'Mean', 'Standard Deviation', 'Confidence Interval',
                                    'Minimum value', 'Maximum value']
                config.roi_stat_number = roi_stat_options.index(config.roi_stat_number)

                conf_level_options = ['80%, 1.28', '85%, 1.44', '90%, 1.64', '95%, 1.96', '98%, 2.33', '99%, 2.58']
                config.bootstrap_alpha = float(f"0.{re.split('%', config.conf_level_number)[0]}")
                config.conf_level_number = conf_level_options.index(config.conf_level_number)

                brain_plot_opts = ['Mean', 'Mean (within roi scaled)', 'Mean (mixed roi scaled)',
                                   'Produce all three figures']
                config.brain_fig_file = brain_plot_opts.index(config.brain_fig_file)

                config.parameter_dict = {config.parameter_dict1[i]:
                                             config.parameter_dict2[i] for i in range(len(config.parameter_dict1))}

                return config

            except (toml.decoder.TomlDecodeError, AttributeError):
                raise Exception('Config file not in correct format or missing entries.')

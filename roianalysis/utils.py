import argparse
import os
import shutil
from glob import glob
from tkinter import Tk, filedialog
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import multiprocess as mp
from scipy.sparse import csr_matrix

from roianalysis import config


class Utils:
    @staticmethod
    def argparser():
        # Create the parser
        parser = argparse.ArgumentParser(prog='roi_analysis_tool',
                                         description='Convert voxelwise statistics to regionwise statistics for MRI data.')

        # Add the arguments
        parser.add_argument('--brain_loc', dest='brain_loc', action='store', type=str,
                            help='Directory location of brain files for analysis. '
                                 'Can be set in config.py to use a GUI to find folder instead).')

        parser.add_argument('--json_loc', dest='json_loc', action='store', type=str,
                            help='Directory location of json files produced by the roi_analysis_tool '
                                 '(can be set in config.py to use a GUI to find folder instead).')

        parser.add_argument('--make_table', dest='make_table', action='store_true',
                            help='Use this flag to create a csv file to store parameter information about files.'
                                 'Recommended that this file is created and filled in before tool execution'
                                 ' (this setting can alternatively be set to True or False in config.py).')

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
    def file_browser(chdir=False):
        root = Tk()  # Create tkinter window

        root.withdraw()  # Hide tkinter window
        root.update()

        directory = filedialog.askdirectory()

        root.update()
        root.destroy()  # Destroy tkinter window

        if chdir:
            os.chdir(directory)

        if config.verbose:
            print(f"Selected directory: {directory}")

        return directory

    @staticmethod
    def save_config(directory):
        with open(directory + '/config_log.py', 'w') as f:
            with open('roianalysis/config.py', 'r') as r:
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
                print(f"Starting processing pool using {workers} cores.")
            return ctx.Pool(processes=workers)

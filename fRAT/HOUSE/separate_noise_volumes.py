from functools import reduce

import numpy as np
from scipy.stats import stats

from fRAT.utils import Utils

UTILITY_NAME = "Separate noise volumes"
FOLDER_NAME = ["noise_volume", "func_volumes"]


def run(*args, **kwargs):
    SeparateNoiseVolumes(*args, **kwargs)


class SeparateNoiseVolumes:
    def __init__(self, *args, **kwargs):
        self.config = args[0]
        self.file = args[1]
        self.no_ext_file = args[2]
        self.project_root = args[3]
        self.participant_name = args[4]
        self.participant_dir = args[5]
        self.output_folder = args[6]

        self.separate_noise_from_func()

    def separate_noise_from_func(self):
        noise_data = None
        func_data = None

        data, header = Utils.load_brain(self.file)

        mean_of_timepoints = data.reshape(reduce(lambda x, y: x*y, data.shape[0:3]), data.shape[3]).mean(axis=0)
        outlier = np.where(np.abs(stats.zscore(mean_of_timepoints)) > 3)

        if len(outlier[0]) == 1 and (outlier[0][0] in (0, data.shape[3] - 1)): # todo hook new folder up to isnr calc
            noise_data = data[:, :, :, outlier]
            func_data = np.delete(data, outlier, axis=3)

        elif len(outlier[0]) == 0:
            print(f'        No noise volume found for file: {self.no_ext_file}')

        else:
            raise Exception(f'Incorrect or multiple potential noise volumes found for file: {self.no_ext_file}.')

        if self.config.verbose and outlier[0][0] == 0:
            print(f'            Noise volume found at beginning of timeseries for file: {self.no_ext_file}')

        elif self.config.verbose and outlier[0][0] == data.shape[3] - 1:
            print(f'            Noise volume found at end of timeseries for {self.no_ext_file}')

        Utils.save_brain(noise_data, '_noise_volume', self.no_ext_file,
                         f"{self.participant_dir}/{FOLDER_NAME[0]}", header)

        Utils.save_brain(func_data, '', self.no_ext_file,
                         f"{self.participant_dir}/{FOLDER_NAME[1]}", header)

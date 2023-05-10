import numpy as np
import pandas as pd

from fRAT.utils import Utils

UTILITY_NAME = "Add Gaussian noise"
FOLDER_NAME = "added_noise"


def run(*args, **kwargs):
    AddNoise(*args, **kwargs)


class AddNoise:
    def __init__(self, *args, **kwargs):
        self.config = args[0]
        self.file = args[1]
        self.no_ext_file = args[2]
        self.project_root = args[3]
        self.participant_name = args[4]
        self.participant_dir = args[5]
        self.output_folder = args[6]

        self.data, self.header = Utils.load_brain(self.file)
        self.add_noise_to_file()

    def add_noise_to_file(self):
        # Get noise level for participant
        noise_value_csv = pd.read_csv(f'{self.project_root}/noiseValues.csv')
        participant_noise_level = float(
            noise_value_csv[noise_value_csv['Participant'] == self.participant_name]['Noise over time'])

        # Save initial data with no added noise
        Utils.save_brain(self.data, ext=f'_noiselevel0', no_ext_file=self.no_ext_file,
                         output_folder=f"{self.participant_dir}/{self.output_folder}", header=self.header)

        # Apply noise multiplier to gaussian noise
        for multiplier in self.config.noise_multipliers:
            gaussian_noise = np.random.default_rng().normal(loc=0.0,
                                                            scale=participant_noise_level * multiplier,
                                                            size=self.data.shape)

            noisy_data = self.data + gaussian_noise
            noisy_data[noisy_data < 0] = 0  # Set values below 0 to 0

            Utils.save_brain(noisy_data, ext=f'_noiselevel{multiplier}', no_ext_file=self.no_ext_file,
                             output_folder=f"{self.participant_dir}/{self.output_folder}", header=self.header)

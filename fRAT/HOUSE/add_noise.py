import numpy as np
import pandas as pd

from fRAT.utils import Utils


def add_noise_to_file(config, file, no_ext_file, project_root, participant_name, participant_dir, output_folder):
    # Get noise level for participant
    noise_value_csv = pd.read_csv(f'{project_root}/noiseValues.csv')
    participant_noise_level = float(noise_value_csv[noise_value_csv['Participant'] == participant_name]['Noise over time'])

    # Load brain
    data, header = Utils.load_brain(file)

    # Save initial data with no added noise
    Utils.save_brain(data, ext=f'_noiselevel0', no_ext_file=no_ext_file,
                     output_folder=f"{participant_dir}/{output_folder}", header=header)

    # Apply noise multiplier to gaussian noise
    for multiplier in config.noise_multipliers:
        gaussian_noise = np.random.default_rng().normal(loc=0.0,
                                                        scale=participant_noise_level * multiplier,
                                                        size=data.shape)

        noisy_data = data + gaussian_noise
        noisy_data[noisy_data < 0] = 0  # Set values below 0 to 0

        Utils.save_brain(noisy_data, ext=f'_noiselevel{multiplier}', no_ext_file=no_ext_file,
                         output_folder=f"{participant_dir}/{output_folder}", header=header)

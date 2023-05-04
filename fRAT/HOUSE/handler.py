import itertools
import os
from pathlib import Path

from . import *
from ..utils import Utils

config = None


def file_setup(func):
    file_location = config.input_folder_name

    if config.output_folder_name != 'DEFAULT':
        output_folder = f"statmaps/{config.output_folder_name}"
    elif func == 'Add Gaussian noise':
        output_folder = 'added_noise'
    elif func == 'Add motion':
        output_folder = 'added_motion'

    if config.base_folder in ("", " "):
        print('Select the directory which contains the subject folders.')
        base_sub_location = Utils.file_browser(title='Select the directory which contains the subject folders')

    else:
        base_sub_location = config.base_folder

    # Create dictionary from each participant directory
    participant_dir, _ = Utils.find_participant_dirs(base_sub_location)
    participants = {participant: None for participant in participant_dir}

    for participant in participants:
        # Find all nifti and analyze files
        participants[participant] = Utils.find_files(f"{participant}/{file_location}", "hdr", "nii.gz", "nii")

        # Make directory to save results
        Utils.check_and_make_dir(f"{participant}/{file_location}/{output_folder}", delete_old=True)

    return participants, output_folder, file_location


def utility_handler(participants, output_folder, file_location, func):
    if config.multicore_processing:
        pool = Utils.start_processing_pool()
    else:
        pool = None

    if config.verbose:
        print(f'\nSaving output in directory: {config.input_folder_name}/{output_folder}')

    for participant, files in participants.items():
        participant_dir = f"{participant}/{file_location}"

        if config.verbose:
            print(f"\nRunning '{func}' utility on participant: {participant_dir.split('/')[-2]}"
                  f"\n      Running utility on {len(files)} files")

        iterable = zip(files,
                       itertools.repeat(participant_dir),
                       itertools.repeat(output_folder),
                       itertools.repeat(func),
                       itertools.repeat(config))

        if config.multicore_processing:
            list(pool.starmap(run_utility, iterable))

        else:
            list(itertools.starmap(run_utility, iterable))

    if config.multicore_processing:
        Utils.join_processing_pool(pool, restart=False)


def run_utility(file, participant_dir, output_folder, func, config):
    no_ext_file = Utils.strip_ext(file)
    file = f"{participant_dir}/{file}"
    base_sub_location = Path(participant_dir).parents[1]
    participant_name = os.path.split(Path(participant_dir).parents[0])[1]

    if config.verbose:
        print(f'        Running utility on file: {no_ext_file}')

    if func == 'Add Gaussian noise':
        add_noise_to_file(config, file, no_ext_file, base_sub_location, participant_name, participant_dir, output_folder)


def HOUSE(cfg, func):
    global config
    config = cfg

    participants, output_folder, file_location = file_setup(func)
    utility_handler(participants, output_folder, file_location, func)

